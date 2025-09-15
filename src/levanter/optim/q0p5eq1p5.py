"""PSGD Q0.5EQ1.5 (Kronecker whitening) optimizer for JAX/Levanter.

This implements the Q0.5EQ1.5 variant of PSGD where the preconditioner update follows:
    dQ = Q^0.5 * E * Q^1.5
with an online orthogonal Procrustes problem solver to keep Q approximately SPD.

Based on the QUAD implementation but with Q0.5EQ1.5 update rule.
"""

import string
from collections import defaultdict
from dataclasses import dataclass
from functools import partial
from typing import Any, Callable, Generic, List, Optional, Tuple, TypeVar, Union, cast

import chex
import jax
import numpy as np
import optax
from jax import numpy as jnp
from jax import vmap
from jax.lax import with_sharding_constraint
from jax.sharding import PartitionSpec
from optax import tree_utils as otu
from optax._src import base, transform
from optax._src.combine import chain
from optax._src.numerics import safe_int32_increment
from optax._src.utils import canonicalize_dtype

from levanter.optim.config import OptimizerConfig


# Define type variables for the pytree structure
T = TypeVar("T")
PartitionSpecTree = TypeVar(
    "PartitionSpecTree", bound=Union[PartitionSpec, List[PartitionSpec], Tuple[PartitionSpec, ...], dict, list, tuple]
)


@OptimizerConfig.register_subclass("q0p5eq1p5")
@dataclass(frozen=True)
class Q0p5EQ1p5Config(OptimizerConfig, Generic[PartitionSpecTree]):
    """Configuration for PSGD-Q0.5EQ1.5 optimizer.

    Notes:
        - LR is usually 3x smaller than adam, weight decay at least 3x larger.
        - Uses Q^0.5 * E * Q^1.5 update rule with Procrustes step for SPD constraint.

    Attributes:
        beta1: Momentum parameter. 0.9 or 0.95 are common values.
        weight_decay: Weight decay coefficient.
        max_grad_norm: Optional gradient norm clipping value.
        max_size_dense: Dimensions larger than this will have diagonal preconditioners,
            otherwise dense.
        max_skew_dense: Dimensions with skew larger than this compared to the other
            dimension will have diagonal preconditioners, otherwise dense.
        preconditioner_lr: Learning rate for preconditioner.
        preconditioner_init_scale: Scale for preconditioner initialization.
        mu_dtype: Dtype of the momentum buffer. Defaults to same dtype as parameters.
        precond_dtype: Dtype of the preconditioners. Defaults to 'float32'.
        scanned_layers: Tree of booleans same structure as params indicating scanned dimensions
            for each layer. PSGD will vmap over leading dimension.
        lax_map_scanned_layers: Whether to use lax.map for scanned layers instead of vmap.
            Useful to save memory with large models.
        lax_map_batch_size: Batch size for lax.map, see JAX docs for more info.
        merge_small_dims: Whether to merge small dimensions to improve preconditioner efficiency.
        target_merged_dim_size: Target size of merged dimensions.
        partition_grads_into_blocks: Whether to partition grads into chunks of size block_size
            for efficiency.
        block_size: Block size to use for partitioning grads.
        params_sharding: Pytree same structure as params of jax.sharding.PartitionSpec.
        preconditioner_sharding: PartitionSpec for preconditioner matrices. Best practice is to
            shard first dimension across fsdp-like mesh axis, or largest/most common axis in params.
            Example: PartitionSpec('fsdp') or PartitionSpec('fsdp', 'tp').
    """

    # some of these are changed from quad defaults to better suit levanter
    lr_style: str | None = "adam"
    beta1: float = 0.95
    weight_decay: float = 0.5
    max_grad_norm: Optional[float] = None
    normalize_grads: bool = False
    max_size_dense: int = 8192
    max_skew_dense: float = 1.0
    preconditioner_lr: float = 0.7
    preconditioner_init_scale: float = 1.0
    procrustes_interval: int = 10  # Apply Procrustes step every N updates
    mu_dtype: Optional[Union[str, jnp.dtype]] = jnp.bfloat16
    precond_dtype: Optional[Union[str, jnp.dtype]] = jnp.bfloat16
    lax_map_scanned_layers: bool = False
    lax_map_batch_size: int = 8
    merge_small_dims: bool = False
    target_merged_dim_size: int = 4096
    partition_grads_into_blocks: bool = False
    block_size: int = 512
    params_sharding: Optional[PartitionSpecTree] = None
    preconditioner_sharding: Optional[tuple[str | None, str | None]] = None

    def build(self, num_train_steps):
        """Creates the optimizer."""

        def _optimizer(learning_rate) -> optax.GradientTransformation:
            precond_partition_spec = (
                PartitionSpec(*self.preconditioner_sharding) if self.preconditioner_sharding is not None else None
            )
            components = []
            if self.max_grad_norm:
                components.append(optax.clip_by_global_norm(self.max_grad_norm))
            components.append(
                scale_by_q0p5eq1p5(
                    lr_style=self.lr_style,
                    b1=self.beta1,
                    normalize_grads=self.normalize_grads,
                    max_size_dense=self.max_size_dense,
                    max_skew_dense=self.max_skew_dense,
                    preconditioner_lr=self.preconditioner_lr,
                    preconditioner_init_scale=self.preconditioner_init_scale,
                    procrustes_interval=self.procrustes_interval,
                    mu_dtype=self.mu_dtype,
                    precond_dtype=self.precond_dtype,
                    lax_map_scanned_layers=self.lax_map_scanned_layers,
                    lax_map_batch_size=self.lax_map_batch_size,
                    merge_small_dims=self.merge_small_dims,
                    target_merged_dim_size=self.target_merged_dim_size,
                    partition_grads_into_blocks=self.partition_grads_into_blocks,
                    block_size=self.block_size,
                    params_sharding=self.params_sharding,
                    preconditioner_sharding=precond_partition_spec,
                )
            )
            if self.weight_decay > 0:
                components.append(optax.add_decayed_weights(self.weight_decay, self.build_weight_decay_mask()))
            components.append(optax.scale_by_learning_rate(learning_rate))
            return optax.chain(*components)

        return optax.inject_hyperparams(_optimizer)(learning_rate=self.lr_scheduler(num_train_steps))


# Import the helper functions and classes from quad.py
from levanter.optim.quad import (
    _get_preconditioner_types,
    _init_Q_exprs,
    get_precond_lr,
    _norm_lower_bound,
    _safe_sharding_constraint,
    _map_fn,
    BlockPartitioner,
    _merge_small_dims,
    _pad_and_stack_matrices,
    _unstack_and_unpad_matrices,
)

try:
    import flax.linen as nn

    have_flax = True
except ImportError:
    have_flax = False
try:
    import haliax as hax

    have_hax = True
except ImportError:
    have_hax = False


def procrustes_step(Q, max_step_size=0.2):
    """
    An in-place (update Q directly) online solver for the orthogonal Procrustes problem,
        min_U || U Q - I ||_F,   s.t. U^H U = I
    by rotating Q as exp(a R) Q, where R = Q^H - Q is the generator and a ||R|| < 1.
    
    Note that such rotations do not include reflections, and thus cannot make real Q SPD if det(Q) < 0.
    """
    R = Q.T - Q  # Q^H in JAX is Q.T for real matrices, Q.conj().T for complex
    max_abs = jnp.max(jnp.abs(R))
    
    def update_q():
        R_normalized = R / max_abs  # normalize R as typically it's too small
        RQ = R_normalized @ Q
        tr_RQ = jnp.sum(jnp.diag(RQ).real)  # trace(RQ)
        
        def rotate_q():
            # rotate Q as exp(a R) Q ~ (I + a R + a^2 R^2/2) Q with an optimal a
            a = max_step_size / _norm_lower_bound(R_normalized)
            RRQ = R_normalized @ RQ
            tr_RRQ = jnp.sum(jnp.diag(RRQ).real)
            
            # the max step size could over-shoot in this case
            a = jax.lax.cond(
                tr_RRQ < 0,
                lambda: jnp.minimum(a, -tr_RQ / tr_RRQ),
                lambda: a
            )
            
            return Q + a * (RQ + 0.5 * a * RRQ)
        
        # only rotate if tr_RQ > 0, otherwise Q is already Hermitian
        return jax.lax.cond(
            tr_RQ > 0,
            rotate_q,
            lambda: Q
        )
    
    # only update if max_abs is not too small (to avoid division by zero/subnormal)
    return jax.lax.cond(
        max_abs > jnp.finfo(Q.dtype).smallest_normal,
        update_q,
        lambda: Q
    )


def _update_precond_q0p5eq1p5(Q, L, G, key, step, exprs, precond_lr, procrustes_interval):
    """Update Q using Q0.5EQ1.5 method and return preconditioned gradient.
    
    This implements the update: dQ = Q^0.5 * E * Q^1.5
    where E is computed from the gradient information.
    Procrustes step is applied every procrustes_interval updates.
    """
    exprP, exprGs = exprs

    # Add tiny numerical stability noise
    Pg = jnp.einsum(exprP, *Q, *Q, G + jax.random.normal(key, G.shape, G.dtype) * 1e-8)
    
    total_numel = G.size
    betaL = 0.95
    
    def _update_single_q_l(i, q, l):
        term1 = jnp.einsum(exprGs[i], Pg, Pg)
        
        if q.ndim < 2:  # diagonal or scalar Q
            term2 = total_numel / q.size
            ell = jnp.max(term1) + term2
            l_new = jnp.maximum(betaL * l + (1 - betaL) * ell, ell)
            lr_over_l = (precond_lr / l_new).astype(q.dtype)
            q_new = q * (1 - lr_over_l * (term1 - term2))
        else:  # matrix Q
            term2 = total_numel / q.shape[0]
            ell = _norm_lower_bound(term1) + term2
            l_new = jnp.maximum(betaL * l + (1 - betaL) * ell, ell)
            lr_over_l = (precond_lr / l_new).astype(q.dtype)
            # Q0.5EQ1.5 update: q = q - lr * (term1 @ q - term2 * q)
            q_new = q - lr_over_l * (term1 @ q - term2 * q)
            # Apply Procrustes step conditionally to maintain SPD property
            q_new = jax.lax.cond(
                step % procrustes_interval == 0,
                lambda q: procrustes_step(q),
                lambda q: q,
                q_new
            )
            
        return q_new, l_new

    Q_L_new = [_update_single_q_l(i, q, l) for i, (q, l) in enumerate(zip(Q, L))]
    Q_new = [ql[0] for ql in Q_L_new]
    L_new = [ql[1] for ql in Q_L_new]
    
    # Recompute Pg with updated Q_new
    Pg_new = jnp.einsum(exprP, *Q_new, *Q_new, G)

    return Q_new, L_new, Pg_new


def scale_by_q0p5eq1p5(
    lr_style: str | None = "adam",
    b1: float = 0.95,
    normalize_grads: bool = False,
    max_size_dense: int = 8192,
    max_skew_dense: float = 1.0,
    preconditioner_lr: float = 0.7,
    preconditioner_init_scale: float = 1.0,
    procrustes_interval: int = 10,
    mu_dtype: Optional[Union[str, jnp.dtype]] = None,
    precond_dtype: Optional[Union[str, jnp.dtype]] = None,
    scanned_layers: Optional[base.Params] = None,
    lax_map_scanned_layers: bool = False,
    lax_map_batch_size: int = 8,
    merge_small_dims: bool = False,
    target_merged_dim_size: int = 8192,
    partition_grads_into_blocks: bool = False,
    block_size: int = 512,
    params_sharding: Optional[PartitionSpecTree] = None,
    preconditioner_sharding: Optional[tuple[str | None, str | None]] = None,
    **kwargs,
) -> base.GradientTransformation:
    """
    Implements PSGD-Q0.5EQ1.5 Kronecker whitening optimizer.
    
    This is based on the QUAD implementation but uses the Q0.5EQ1.5 update rule
    with Procrustes step for maintaining SPD property.
    
    Args:
        Same as scale_by_quad but uses Q0.5EQ1.5 update rule.
    
    Returns:
        optax.GradientTransformation
    """
    mu_dtype = canonicalize_dtype(mu_dtype)
    precond_dtype = canonicalize_dtype(precond_dtype or jnp.float32)
    lax_map = lax_map_scanned_layers
    bs = lax_map_batch_size
    scanned_layers = None

    def init_fn(params, return_partition_specs_only=False):
        # This is identical to the QUAD init_fn
        # unbox if haliax style partitioned
        scanned_layers_ = None
        params_sharding_ = params_sharding
        if have_hax:
            if any(
                isinstance(x, hax.NamedArray)
                for x in jax.tree.leaves(params, is_leaf=lambda x: isinstance(x, hax.NamedArray))
            ):
                # if in haliax, we can grab scanned_layers and params_sharding from params
                # this does not support nested stacks
                if scanned_layers_ is None:
                    scanned_layers_ = jax.tree.map(
                        lambda x: (jax.tree.map(lambda _: True, x) if isinstance(x, hax.nn.Stacked) else False),
                        params,
                        is_leaf=lambda x: isinstance(x, hax.nn.Stacked),
                    )
                if params_sharding_ is None:
                    try:
                        params_sharding_ = hax.partitioning.infer_resource_partitions(params)
                        params_sharding_ = jax.tree.map(lambda x: x.spec, params_sharding_)
                    except ValueError:
                        # No mesh available, skip sharding
                        params_sharding_ = None
                params, params_struct = jax.tree.flatten(params)
                scanned_layers_ = jax.tree.leaves(scanned_layers_)
                if params_sharding_ is not None:
                    params_sharding_ = jax.tree.leaves(params_sharding_)

        have_params_sharding = params_sharding_ is not None
        have_qs_sharding = have_params_sharding or preconditioner_sharding is not None

        # unbox if flax style partitioned
        if have_flax:
            params = jax.tree.map(
                lambda x: x.unbox() if isinstance(x, nn.Partitioned) else x,
                params,
                is_leaf=lambda x: isinstance(x, nn.Partitioned),
            )

        # check that there is a PartitionSpec for every param
        if params_sharding_ is not None:
            assert len(jax.tree.leaves(params_sharding_)) == len(
                jax.tree.leaves(params)
            ), "There must be a PartitionSpec for every parameter in PSGD-Q0.5EQ1.5."
        # check that preconditioner sharding length is at least 1
        if preconditioner_sharding is not None:
            assert len(preconditioner_sharding) > 0, (
                "preconditioner_sharding must have length > 0. For example, "
                "PartitionSpec(None) or PartitionSpec('fsdp', None) are valid."
            )

        # extend partition specs
        if have_params_sharding:
            params_sharding_ = jax.tree.map(
                lambda p, sh: PartitionSpec(*(sh + (None,) * (len(p.shape) - len(sh)))),
                params,
                params_sharding_,
            )
        preconditioner_sharding_ = preconditioner_sharding
        if preconditioner_sharding is not None:
            if len(preconditioner_sharding) < 2:
                preconditioner_sharding_ = PartitionSpec(preconditioner_sharding[0], None)

        # reshape params shaped () to (1,) to make things simpler
        params = jax.tree.map(lambda p: p[None] if len(p.shape) == 0 else p, params)
        if have_params_sharding:
            params_sharding_ = jax.tree.map(
                lambda sh: PartitionSpec(None) if sh == PartitionSpec() else sh,
                params_sharding_,
            )

        # scanned layers
        if scanned_layers_ is None:
            scanned_layers_ = jax.tree.map(lambda _: False, params)
        scanned_sizes = jax.tree.map(lambda p, s: p.shape[0] if s else 0, params, scanned_layers_)

        # momentum
        mu = None
        mu_sharding = params_sharding_
        if b1 > 0 and not return_partition_specs_only:
            mu = jax.tree.map(lambda x: jnp.zeros_like(x, dtype=mu_dtype), params)
            # apply params sharding to momentum buffer
            if have_params_sharding:
                mu = _safe_sharding_constraint(mu, params_sharding_)

        # which preconditioners will be diagonal
        dim_diag = jax.tree.map(
            lambda p, s: _get_preconditioner_types(
                p.shape[int(s) :],
                max_size_dense,
                max_skew_dense,
            ),
            params,
            scanned_layers_,
        )

        # split sharding specs
        scanned_dim_sharding = None
        sharding_without_scan = None
        if have_params_sharding:
            scanned_dim_sharding = jax.tree.map(
                lambda sh, s: PartitionSpec(sh[0]) if s else None,
                params_sharding_,
                scanned_layers_,
            )
            sharding_without_scan = jax.tree.map(
                lambda sh, s: PartitionSpec(*(sh[int(s) :])),
                params_sharding_,
                scanned_layers_,
            )

        # merge small dimensions
        nones = jax.tree.map(lambda _: None, params)
        merged_shapes = jax.tree.map(lambda p, s: p.shape[int(s) :], params, scanned_layers_)
        if merge_small_dims:
            output = jax.tree.map(
                lambda p, s, dd, sh: _merge_small_dims(p.shape[int(s) :], target_merged_dim_size, dd, sh),
                params,
                scanned_layers_,
                dim_diag,
                sharding_without_scan if have_params_sharding else nones,
            )
            merged_shapes, dim_diag, sharding_without_scan = [
                jax.tree.map(lambda _, x: x[i], params, output) for i in range(3)
            ]

        # partition grads into blocks
        partitioned_shapes = merged_shapes
        if partition_grads_into_blocks:
            partitioners = jax.tree.map(
                lambda _, ps, dd: BlockPartitioner(ps, block_size, dd),
                params,
                merged_shapes,
                dim_diag,
            )
            # we can grab resulting shapes from partitioners
            partitioned_shapes = jax.tree.map(lambda _, p_cls: p_cls._padded_stacked_shape, params, partitioners)

        # initialize preconditioners
        output = jax.tree.map(
            lambda _, ps, dd, sh: list(
                _init_Q_exprs(
                    ps[1:] if partition_grads_into_blocks else ps,
                    preconditioner_init_scale,
                    dd,
                    precond_dtype,
                    existing_Q=True if return_partition_specs_only else None,
                    precond_sharding=preconditioner_sharding_,
                    param_sharding=sh,
                )
            ),
            params,
            partitioned_shapes,
            dim_diag,
            sharding_without_scan if have_params_sharding else nones,
        )
        if return_partition_specs_only:
            exprs, Qs_sharding_no_leading_dims = [jax.tree.map(lambda _, x: x[i], params, output) for i in range(2)]
        else:
            Qs, Ls, exprs, Qs_sharding_no_leading_dims = [
                jax.tree.map(lambda _, x: x[i], params, output) for i in range(4)
            ]
        Qs_sharding = None
        if have_qs_sharding:
            # add scan and stack dims to Qs sharding
            def add_dims_to_spec(_, qss, sds):
                if partition_grads_into_blocks:
                    qss = jax.tree.map(lambda qs: PartitionSpec(*((None,) + qs)), qss)
                if sds is not None:
                    qss = jax.tree.map(lambda qs: PartitionSpec(*(sds + qs)), qss)
                return qss

            Qs_sharding = jax.tree.map(
                add_dims_to_spec,
                params,
                Qs_sharding_no_leading_dims,
                scanned_dim_sharding,
            )

        if not return_partition_specs_only:
            # broadcast Qs and Ls for stacks and scans
            def broadcast_qs(_, ps, x, s):
                stack_n = ps[0]
                if partition_grads_into_blocks:
                    # add leading dim for stacked partitions
                    x = jax.tree.map(lambda x: jnp.repeat(jnp.expand_dims(x, 0), stack_n, axis=0), x)
                if s > 0:
                    # add leading dim if we're scanning this layer
                    x = jax.tree.map(lambda d: jnp.repeat(jnp.expand_dims(d, 0), s, axis=0), x)
                return x

            Qs = jax.tree.map(broadcast_qs, params, partitioned_shapes, Qs, scanned_sizes)
            Ls = jax.tree.map(broadcast_qs, params, partitioned_shapes, Ls, scanned_sizes)
            if have_qs_sharding:
                Qs = _safe_sharding_constraint(Qs, Qs_sharding)

        if return_partition_specs_only:
            return dict(
                count=PartitionSpec(),
                mu=mu_sharding,
                Qs_preconditioners=Qs_sharding,
                Ls_lipschitz=PartitionSpec(None),
            )

        return dict(
            count=jnp.zeros([], jnp.int32),
            mu=mu,
            Qs_preconditioners=Qs,
            Ls_lipschitz=Ls,
        )

    def update_fn(updates: base.Updates, state: dict, params: base.Params = None):
        del params
        count_inc = safe_int32_increment(state["count"])
        precond_lr_t = get_precond_lr(preconditioner_lr, count_inc)

        # unbox if haliax style partitioned
        scanned_layers_ = scanned_layers
        params_sharding_ = params_sharding
        hax_partitioned = False
        if have_hax:
            if any(
                isinstance(x, hax.NamedArray)
                for x in jax.tree.leaves(updates, is_leaf=lambda x: isinstance(x, hax.NamedArray))
            ):
                hax_partitioned = True
                # if in haliax, we can grab scanned_layers and params_sharding from params
                # this does not support nested stacks
                if scanned_layers_ is None:
                    scanned_layers_ = jax.tree.map(
                        lambda x: (jax.tree.map(lambda _: True, x) if isinstance(x, hax.nn.Stacked) else False),
                        updates,
                        is_leaf=lambda x: isinstance(x, hax.nn.Stacked),
                    )
                if params_sharding_ is None:
                    try:
                        params_sharding_ = hax.partitioning.infer_resource_partitions(updates)
                        params_sharding_ = jax.tree.map(lambda x: x.spec, params_sharding_)
                    except ValueError:
                        # No mesh available, skip sharding
                        params_sharding_ = None
                updates, updates_struct = jax.tree.flatten(updates)
                scanned_layers_ = jax.tree.leaves(scanned_layers_)
                if params_sharding_ is not None:
                    params_sharding_ = jax.tree.leaves(params_sharding_)

        have_params_sharding = params_sharding_ is not None
        if have_params_sharding:
            original_params_sharding_ = params_sharding_
        have_qs_sharding = have_params_sharding or preconditioner_sharding is not None

        # unbox if flax style partitioned
        flax_partitioned = False
        if have_flax:
            boxed_updates, grads_structure = jax.tree.flatten(
                updates,
                is_leaf=lambda g: isinstance(g, (chex.Array, nn.Partitioned, jax.ShapeDtypeStruct)),
            )
            if any(isinstance(g, nn.Partitioned) for g in boxed_updates):
                flax_partitioned = True
                updates = [g.unbox() for g in boxed_updates]
                updates = grads_structure.unflatten(updates)

        # extend partition specs
        if have_params_sharding:
            params_sharding_ = jax.tree.map(
                lambda g, sh: PartitionSpec(*(sh + (None,) * (len(g.shape) - len(sh)))),
                updates,
                params_sharding_,
            )
        preconditioner_sharding_ = preconditioner_sharding
        if preconditioner_sharding is not None:
            if len(preconditioner_sharding) < 2:
                preconditioner_sharding_ = PartitionSpec(preconditioner_sharding[0], None)

        # reshape params shaped () to (1,) to make things simpler
        input_shapes = jax.tree.map(lambda g: g.shape, updates)
        updates = jax.tree.map(lambda g: g[None] if len(g.shape) == 0 else g, updates)
        if have_params_sharding:
            params_sharding_ = jax.tree.map(
                lambda sh: PartitionSpec(None) if sh == PartitionSpec() else sh,
                params_sharding_,
            )

        # scanned layers
        if scanned_layers_ is None:
            scanned_layers_ = jax.tree.map(lambda _: False, updates)

        # optionally normalize grads layer-wise
        if normalize_grads:
            updates = jax.tree.map(lambda g: g / (jnp.linalg.norm(g) + 1e-6), updates)

        # momentum
        mu = None
        momentum_updates = updates
        if state["mu"] is not None:
            mu = otu.tree_update_moment(updates, state["mu"], b1, 1)
            if have_params_sharding:
                mu = _safe_sharding_constraint(mu, params_sharding_)
            momentum_updates = otu.tree_bias_correction(mu, b1, count_inc)
        # cast mu back to mu_dtype
        mu = otu.tree_cast(mu, mu_dtype)
        # cast momentum_updates to precond_dtype
        momentum_updates = otu.tree_cast(momentum_updates, precond_dtype)

        # which preconditioners will be diagonal
        dim_diag = jax.tree.map(
            lambda g, s: _get_preconditioner_types(
                g.shape[int(s) :],
                max_size_dense,
                max_skew_dense,
            ),
            momentum_updates,
            scanned_layers_,
        )

        # split sharding specs
        scanned_dim_sharding = None
        sharding_without_scan = None
        if have_params_sharding:
            scanned_dim_sharding = jax.tree.map(
                lambda sh, s: PartitionSpec(sh[0]) if s else None,
                params_sharding_,
                scanned_layers_,
            )
            sharding_without_scan = jax.tree.map(
                lambda sh, s: PartitionSpec(*(sh[int(s) :])),
                params_sharding_,
                scanned_layers_,
            )

        # merge small dimensions
        nones = jax.tree.map(lambda _: None, momentum_updates)
        merged_params_sharding = params_sharding_
        original_shapes = None
        if merge_small_dims:
            original_shapes = jax.tree.map(lambda g, s: g.shape[int(s) :], momentum_updates, scanned_layers_)
            output = jax.tree.map(
                lambda g, dd, s, sh: _merge_small_dims(g.shape[int(s) :], target_merged_dim_size, dd, sh),
                momentum_updates,
                dim_diag,
                scanned_layers_,
                sharding_without_scan if have_params_sharding else nones,
            )
            merged_shapes, dim_diag, sharding_without_scan = [
                jax.tree.map(lambda _, x: x[i], momentum_updates, output) for i in range(3)
            ]
            # reshape
            momentum_updates = jax.tree.map(
                lambda g, s, ns: _map_fn(False, 0, int(s), lambda x, shape=ns: jnp.reshape(x, shape), g),
                momentum_updates,
                scanned_layers_,
                merged_shapes,
            )
            if have_params_sharding:
                # scanned dim sharding + new merged sharding
                merged_params_sharding = jax.tree.map(
                    lambda sws, sds: PartitionSpec(*(sds + sws if sds is not None else sws)),
                    sharding_without_scan,
                    scanned_dim_sharding,
                )
        # constrain sharding
        if have_params_sharding:
            momentum_updates = _safe_sharding_constraint(momentum_updates, merged_params_sharding)

        # partition grads into blocks
        dummy_updates_tree = jax.tree.map(lambda _: jnp.zeros([]), updates)
        n_dims_to_map = jax.tree.map(lambda s: int(s), scanned_layers_)
        partitioned_sharding = merged_params_sharding
        partitioners = None
        partitioned_shapes = None
        if partition_grads_into_blocks:
            partitioners = jax.tree.map(
                lambda g, dd, s: BlockPartitioner(g.shape[int(s) :], block_size, dd),
                momentum_updates,
                dim_diag,
                scanned_layers_,
            )
            # layers become tuples each containing layer's partitions
            momentum_updates = jax.tree.map(
                lambda g, p_cls, s: _map_fn(False, 0, int(s), p_cls.partition, g),
                momentum_updates,
                partitioners,
                scanned_layers_,
            )
            partitioned_shapes = jax.tree.map(
                lambda _, g, s: jax.tree.map(lambda x: x.shape[int(s) :], g),
                dummy_updates_tree,
                momentum_updates,
                scanned_layers_,
            )
            if have_params_sharding:
                # constrain partitions to same sharding as entire layer
                momentum_updates = jax.tree.map(
                    lambda _, g, mps: jax.tree.map(lambda x: _safe_sharding_constraint(x, mps), g),
                    dummy_updates_tree,
                    momentum_updates,
                    merged_params_sharding,
                )
            # pad and stack partitions, tuples become arrays with new leading dim
            momentum_updates = jax.tree.map(
                lambda _, g, s: _map_fn(
                    False,
                    0,
                    int(s),
                    lambda x, bs=block_size: _pad_and_stack_matrices(x, bs),
                    g,
                ),
                dummy_updates_tree,
                momentum_updates,
                scanned_layers_,
            )
            if have_params_sharding:
                # add dim to sharding specs for new stacked dim
                partitioned_sharding = jax.tree.map(
                    lambda mps, s: PartitionSpec(*(mps[: int(s)] + (None,) + mps[1:])),
                    merged_params_sharding,
                    scanned_layers_,
                )
            n_dims_to_map = jax.tree.map(lambda x: x + 1, n_dims_to_map)
        # constrain sharding
        if have_params_sharding:
            momentum_updates = _safe_sharding_constraint(momentum_updates, partitioned_sharding)

        # get einsum expressions and Qs sharding
        Qs = state["Qs_preconditioners"]
        Ls = state["Ls_lipschitz"]
        Qs_sharding = None
        exprs_and_sharding = jax.tree.map(
            lambda g, dd, sh, nm: _init_Q_exprs(
                g.shape[nm:],
                preconditioner_init_scale,
                dd,
                precond_dtype,
                existing_Q=True,
                existing_L=True,
                precond_sharding=preconditioner_sharding_,
                param_sharding=sh,
            ),
            momentum_updates,
            dim_diag,
            sharding_without_scan if have_params_sharding else nones,
            n_dims_to_map,
        )
        exprs, Qs_sharding_no_leading_dims = [
            jax.tree.map(lambda _, x: x[i], dummy_updates_tree, exprs_and_sharding) for i in range(2)
        ]
        Qs_sharding = None
        if have_qs_sharding:
            # add scan and stack dims to Qs sharding
            def add_dims_to_spec(_, qss, sds):
                if partition_grads_into_blocks:
                    qss = jax.tree.map(lambda qs: PartitionSpec(*((None,) + qs)), qss)
                if sds is not None:
                    qss = jax.tree.map(lambda qs: PartitionSpec(*(sds + qs)), qss)
                return qss

            Qs_sharding = jax.tree.map(
                add_dims_to_spec,
                dummy_updates_tree,
                Qs_sharding_no_leading_dims,
                scanned_dim_sharding,
            )

        # balance preconditioners about every 50 updates
        def balance_Qs(Qs_to_bal):
            def _balance_Q(Q):
                norms = jnp.array([jnp.max(jnp.abs(q)) for q in Q], dtype=jnp.float32)
                gmean = jnp.exp(jnp.mean(jnp.log(norms)))
                to_mul = gmean / norms
                return [q * x.astype(q.dtype) for q, x in zip(Q, to_mul)]

            return jax.tree.map(
                lambda _, Q, nm: _map_fn(False, 0, nm, _balance_Q, Q),
                dummy_updates_tree,
                Qs_to_bal,
                n_dims_to_map,
            )

        Qs = jax.lax.cond(count_inc % 100 == 0, balance_Qs, lambda qs: qs, Qs)
        if have_qs_sharding:
            Qs = _safe_sharding_constraint(Qs, Qs_sharding)

        # update Qs with random keys for numerical stability
        key = jax.random.fold_in(jax.random.PRNGKey(42), state["count"])
        flat_updates, momentum_updates_struct = jax.tree.flatten(momentum_updates)
        flat_leaf_keys = jax.random.split(key, len(flat_updates))
        leaf_keys_tree = momentum_updates_struct.unflatten(list(flat_leaf_keys))
        # create per-leaf stacked keys matching mapped leading dims
        def make_keys(k, g, nm):
            nm = int(nm)
            if nm <= 0:
                return k
            num = int(np.prod(g.shape[:nm]))
            ks = jax.random.split(k, num)
            return jnp.reshape(ks, g.shape[:nm] + (2,))

        keys = jax.tree.map(make_keys, leaf_keys_tree, momentum_updates, n_dims_to_map)
        # update Qs and constrain sharding
        with jax.default_matmul_precision("high"):
            # First update preconditioners using Q0.5EQ1.5 method and get Pg
            Qs_Ls_Pg = jax.tree.map(
                lambda g, Q, L, expr, nm, qss, sh, k: _map_fn(
                    lax_map,
                    bs,
                    nm,
                    partial(
                        _update_precond_q0p5eq1p5,  # Use Q0.5EQ1.5 update instead of QUAD
                        step=count_inc,
                        exprs=expr,
                        precond_lr=precond_lr_t,
                        procrustes_interval=procrustes_interval,
                    ),
                    Q,
                    L,
                    g,
                    k,
                ),
                momentum_updates,
                Qs,
                Ls,
                exprs,
                n_dims_to_map,
                Qs_sharding_no_leading_dims if have_qs_sharding else nones,
                sharding_without_scan if have_params_sharding else nones,
                keys,
            )
        Qs, Ls, Pgs = [
            jax.tree_util.tree_map(lambda qlp: qlp[i], Qs_Ls_Pg, is_leaf=lambda x: isinstance(x, tuple))
            for i in range(3)
        ]
        
        # Use the preconditioned gradients we already computed
        precond_gs = Pgs
        if have_qs_sharding:
            Qs = _safe_sharding_constraint(Qs, Qs_sharding)
        if have_params_sharding:
            precond_gs = _safe_sharding_constraint(precond_gs, partitioned_sharding)
        # cast Qs back to precond_dtype
        Qs = otu.tree_cast(Qs, precond_dtype)
        
        # Clip preconditioned gradients to max RMS=1.1 and apply lr style
        precond_gs = jax.tree.map(
            lambda g: g / jnp.maximum(jnp.sqrt(jnp.mean(jnp.square(g))) / 1.1, 1.0), precond_gs
        )
        if lr_style == "adam":
            precond_gs = jax.tree.map(lambda g: g / 5.0, precond_gs)

        # unpartition grads
        if partition_grads_into_blocks:
            precond_gs = jax.tree.map(
                lambda g, s, ps: _map_fn(
                    False,
                    0,
                    int(s),
                    lambda p, shapes=ps: _unstack_and_unpad_matrices(p, shapes),
                    g,
                ),
                precond_gs,
                scanned_layers_,
                partitioned_shapes,
            )
            if have_params_sharding:
                precond_gs = _safe_sharding_constraint(precond_gs, merged_params_sharding)
            precond_gs = jax.tree.map(
                lambda _, g, s, p_cls: _map_fn(False, 0, int(s), p_cls.merge_partitions, g),
                dummy_updates_tree,
                precond_gs,
                scanned_layers_,
                partitioners,
            )
            if have_params_sharding:
                precond_gs = _safe_sharding_constraint(precond_gs, merged_params_sharding)

        # un-merge dimensions
        if merge_small_dims:
            precond_gs = jax.tree.map(
                lambda g, s, os: _map_fn(False, 0, int(s), lambda p, shape=os: jnp.reshape(p, shape), g),
                precond_gs,
                scanned_layers_,
                original_shapes,
            )
            if have_params_sharding:
                precond_gs = _safe_sharding_constraint(precond_gs, params_sharding_)

        # return scalars to original shape
        precond_gs = jax.tree.map(lambda g, s: jnp.reshape(g, s), precond_gs, input_shapes)

        # final constraint for good measure
        if have_params_sharding:
            precond_gs = _safe_sharding_constraint(precond_gs, original_params_sharding_)

        # box preconditioned grads
        if flax_partitioned:
            flat_precond_gs, _ = jax.tree.flatten(precond_gs)
            precond_gs = [bu.replace_boxed(g) for bu, g in zip(boxed_updates, flat_precond_gs)]
            precond_gs = grads_structure.unflatten(precond_gs)
        if hax_partitioned:
            precond_gs = updates_struct.unflatten(precond_gs)

        # dtypes and new state
        mu = otu.tree_cast(mu, mu_dtype)
        Qs = otu.tree_cast(Qs, precond_dtype)
        Ls = otu.tree_cast(Ls, jnp.float32)
        state = dict(
            count=count_inc,
            mu=mu,
            Qs_preconditioners=Qs,
            Ls_lipschitz=Ls,
        )

        return precond_gs, state

    return base.GradientTransformation(init_fn, update_fn)

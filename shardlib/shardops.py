import shardlib.shardtypes as shardtypes
from jax import lax
import jax.numpy as jnp
import jax

def all_gather(spec: str, x):
  """String-specified all-gather operation.
  
  For example:
    all_gather('A/x/y B/z C/w -> A B C/w', x)
  """
  before, after = spec.split('->')
  before = shardtypes.ShapeSpec.parse(before)
  after = shardtypes.ShapeSpec.parse(after)
  shardtypes.check(x.dtype, before, x)
  for i, (before_dim, after_dim) in enumerate(zip(before.dims, after.dims)):
    # Check that after_dim.sharding is a prefix of before_dim.sharding
    after_n = len(after_dim.sharding)
    if before_dim.shape != after_dim.shape or before_dim.sharding[:after_n] != after_dim.sharding:
      raise ValueError(f'Cannot all-gather {before_dim} into {after_dim}')
    if len(before_dim.sharding) == after_n:
      continue
    x = lax.all_gather(x, tuple(before_dim.sharding[after_n:]), axis=i, tiled=True)
  shardtypes.check(x.dtype, after, x)
  return x
  
def psum_scatter(spec: str, x):
  """String-specified reduce-scatter operation.
  
  For example:
    psum_scatter('A B C/w -> A/x/y B/z C/w', x)
  """
  before, after = spec.split('->')
  before = shardtypes.ShapeSpec.parse(before)
  after = shardtypes.ShapeSpec.parse(after)
  shardtypes.check(x.dtype, before, x)
  for i, (before_dim, after_dim) in enumerate(zip(before.dims, after.dims)):
    # Check that before_dim.sharding is a prefix of after_dim.sharding
    before_n = len(before_dim.sharding)
    if before_dim.shape != after_dim.shape or after_dim.sharding[:before_n] != before_dim.sharding:
      raise ValueError(f'Cannot reduce-scatter {before_dim} into {after_dim}')
    if len(after_dim.sharding) == before_n:
      continue
    x = lax.psum_scatter(x, tuple(after_dim.sharding[before_n:]), scatter_dimension=i, tiled=True)
  shardtypes.check(x.dtype, after, x)
  return x

def einsum_unreduced(spec: str, x, y, **kwargs):
  """Ordinary chip-local einsum, but with sharding-aware typechecking.
  
  Note that this function does not do any chip-to-chip communication. If the inputs are
  sharded over the contraction dimensions, the caller is responsible for reducing the result
  over those dimensions. For example:

    c = einsum_unreduced('A/x B/y, B/y C/z -> A/x/z', a, b)
    # c still needs to be reduced over the y axis.
    d = psum_scatter('A/x/z -> A/x/z/y', c)
    # Now the post-einsum reduction is complete.
  """
  tmp, result = spec.split('->')
  lhs, rhs = tmp.split(',')
  lhs = shardtypes.ShapeSpec.parse(lhs)
  rhs = shardtypes.ShapeSpec.parse(rhs)
  result = shardtypes.ShapeSpec.parse(result)
  shardtypes.check(x.dtype, lhs, x)
  shardtypes.check(y.dtype, rhs, y)
  # Convert to jax einsum syntax, with single-letter variables.
  jaxspec = ''

  vars = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'
  var_i = 0
  dim_table = {}
  def map_var(dim):
    if dim in dim_table:
      return dim_table[dim]
    nonlocal var_i
    if var_i >= len(vars):
      raise ValueError('Too many dimensions in einsum, we ran out of variables')
    var = vars[var_i]
    var_i += 1
    dim_table[dim] = var
    return var

  for dim in lhs.dims:
    jaxspec += map_var(dim)
  jaxspec += ','
  for dim in rhs.dims:
    jaxspec += map_var(dim)
  jaxspec += '->'
  for dim in result.dims:
    jaxspec += map_var(dim)
  r = jnp.einsum(jaxspec, x, y, **kwargs)
  shardtypes.check(r.dtype, result, r)
  return r

def index_unreduced(spec: str, table, indices):
  """String-specified sharded table lookup operation.
  
  For example:
    index_unreduced(table, indices, 'A [B/x/y] C/z, D/w A -> C/z A D/w')
  
  In this example, the integers in `indices` are used as lookup addresses into the
  `B` dimension of `table`, and all other dimensions (`A`, `C`, `D`) are vmapped over.
  
  This operation does not do any chip-to-chip communication, even though the table
  may be sharded. If the axis inside square brackets is sharded, corresponding to
  different table indices on different shards, a table lookup will be performed on each
  shard, but only one shard will return a nonzero result: the other shards, where the
  index is out of bounds, will return zero. The caller is required to reduce the output
  over the axes specified by the square brackets: in the above example, the caller must
  reduce over `x` and `y` axes.
  """
  tmp, result = spec.split('->')
  lhs, rhs = tmp.split(',')
  lhs_dims = lhs.split()
  index_axis = None
  for i, dim in enumerate(lhs_dims):
    if dim.startswith('['):
      index_axis = i
      if not dim.endswith(']'):
        raise ValueError(f'Expected closing bracket in {dim}')
      lhs_dims[i] = dim[1:-1]
      break
  if index_axis is None:
    raise ValueError(f'Expected an index axis in {lhs}')
  
  lhs_dims = [shardtypes.DimSpec.parse(dim) for dim in lhs_dims]
  lhs_spec = shardtypes.ShapeSpec(lhs_dims)
  rhs_spec = shardtypes.ShapeSpec.parse(rhs)
  result_spec = shardtypes.ShapeSpec.parse(result)
  shardtypes.check(table.dtype, lhs_spec, table)
  shardtypes.check(indices.dtype, rhs_spec, indices)

  # Do the base operation on scalars, then do a sequence of vmap operations to bring it up
  # to the desired shape.
  def base_op(table, index):
    len_per_chip = table.shape[0]
    lower_bound = len_per_chip * lax.axis_index(lhs_dims[index_axis].sharding)
    upper_bound = lower_bound + len_per_chip
    in_bounds = (lower_bound <= index) & (index < upper_bound)
    return jnp.where(in_bounds, table[jnp.where(in_bounds, index - lower_bound, 0)], 0)
  
  op = base_op

  lhs_dims_handled = [False] * len(lhs_dims)
  lhs_dims_handled[index_axis] = True
  rhs_dims_handled = [False] * len(rhs_spec.dims)
  for dim in reversed(result_spec.dims):
    try:
      lhs_index = lhs_dims.index(dim)
      lhs_vmap_axis = sum(lhs_dims_handled[:lhs_index])
      assert not lhs_dims_handled[lhs_index]
      lhs_dims_handled[lhs_index] = True
    except ValueError:
      lhs_index = None
      lhs_vmap_axis = None

    try:
      rhs_index = rhs_spec.dims.index(dim)
      rhs_vmap_axis = sum(rhs_dims_handled[:rhs_index])
      assert not rhs_dims_handled[rhs_index]
      rhs_dims_handled[rhs_index] = True
    except ValueError:
      rhs_index = None
      rhs_vmap_axis = None
    
    op = jax.vmap(op, in_axes=(lhs_vmap_axis, rhs_vmap_axis), out_axes=0)
  
  assert all(lhs_dims_handled)
  assert all(rhs_dims_handled)

  result = op(table, indices)
  shardtypes.check(result.dtype, result_spec, result)
  return result

def axis_size(name: str) -> int:
  """Return the size of the axis with the given name."""
  return jax.lax.psum(1, name)
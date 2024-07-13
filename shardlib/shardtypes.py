"""Type annotations for JAX arrays with sharding information.

# Shape checking

Example:

```
import jax
shardtypes.register_with_typeguard()
from shardlib.shardtypes import f32
from typeguard import typechecked

@typechecked
def center_channels(x: f32[b'batch/d channels']) -> f32[b'batch/d channels']:
  return x - jax.numpy.mean(x, axis=-1, keepdims=True)
```

The type syntax is `<dtype>[<shape string>]`, where `dtype` is imported from `shardlib.shardtypes`,
and `<shape string>` is a space-separated list of dimensions. Each dimension consists of a dimension
name (e.g. `batch`), optionally followed by slashes and sharding axis names, e.g. `batch/d` indicates
that the `batch` tensor dimension is sharded over the `d` device axis. Sharding over multiple axes
is indicated by multiple axis names, e.g. `batch/d/e`.

The shape string may be either a string ('foo') or a bytes object (b'foo'). Strings have special
meaning in Python type annotations (they are used for forward references, and are eval'ed by typeguard),
so the bytes object b'foo' is a workaround to prevent this eval'ing.

Shape checking proceeds by maintaining a table of the sizes of all dimension names in a context 
variable, known as the shape checking scope. The first time a dimension name is encountered,
its size is recorded in the current scope. Subsequent uses of the same dimension name must have
the same size. Device axes (e.g. `/d`) are looked up in the currently configured JAX device mesh,
to determine the size of the axis.

For calls into functions or libraries, it can be useful to clear the shape checking scope, so caller
and callee can use the same variable name to mean different things. This can be done with the `@scope`
function decorator or the `with Scope():` context manager.

# Using type annotations

In addition to driving shape checking, type annotations can be used to drive sharding in JAX functions.
See for example `typed_shard_map`, which is a simplification of JAX's `shard_map` by taking advantage
of sharding in type signatures. 
"""
import inspect
import typing
from collections.abc import Sequence
from contextvars import ContextVar
from enum import IntEnum
from typing import Any, Union
from typing import get_args, get_origin
from typeguard import check_type_internal, typechecked
import jax
import jax.numpy as jnp
from types import GenericAlias
from typeguard import TypeCheckError, TypeCheckerCallable
import dataclasses
from dataclasses import dataclass, make_dataclass
from typeguard import checker_lookup_functions


#### State
# ContextVar(dict[str, int])
_VARS = ContextVar('shardtypes._VARS', default={})

class Scope:
  """Context manager that clears the shape checking scope."""
  def __enter__(self):
    self.token = _VARS.set({})
  
  def __exit__(self, type, value, traceback):
    _VARS.reset(self.token)

def scope(f):
  """Function decorator that clears the shape checking scope."""
  def wrapper(*args, **kwargs):
    with Scope():
      return f(*args, **kwargs)
  return wrapper


def check_size(name: str, size: int):
  """Checks that a dimension has the expected size."""
  try:
    value = int(name)
    if value != size:
      raise TypeCheckError(f'explicit dimension {value}: actually was {size}')
  except ValueError:
    v = _VARS.get()
    if name in v:
      if v[name] != size:
        raise TypeCheckError(f'dimension {name}: expected {v[name]}, got {size}')
    else:
      v[name] = size


#### Shape specs
@dataclass(frozen=True)
class DimSpec:
  """Parsed result of a dimension in a shape string."""
  shape: str
  sharding: Sequence[str]

  @staticmethod
  def parse(spec: str) -> 'DimSpec':
    pieces = spec.split('/')
    shape = pieces[0]
    sharding = tuple(pieces[1:])
    return DimSpec(shape, sharding)

  def __str__(self):
    return '/'.join([self.shape] + list(self.sharding))

@dataclass
class ShapeSpec:
  """Parsed result of a shape string."""
  dims: Sequence[DimSpec]

  @staticmethod
  def parse(spec: Union[bytes, str]) -> 'ShapeSpec':
    if isinstance(spec, bytes):
      spec = spec.decode('utf-8')
    if not isinstance(spec, str):
      print(spec)
      raise ValueError('Expected a string')
    dims = spec.split()  # Split on spaces, trimming excess space
    result = []
    for dim in dims:
      result.append(DimSpec.parse(dim))
    return ShapeSpec(result)

  def partition_spec(self) -> jax.sharding.PartitionSpec:
    result = []
    for dim_spec in self.dims:
      if len(dim_spec.sharding) == 0:
        result.append(None)
      elif len(dim_spec.sharding) == 1:
        result.append(dim_spec.sharding[0])
      else:
        result.append(tuple(dim_spec.sharding))
    return jax.sharding.PartitionSpec(*result)
  
  def __str__(self):
    return ' '.join(str(dim) for dim in self.dims)

#### Shape checking
def _partition_spec_equiv(lhs: jax.sharding.PartitionSpec, rhs: jax.sharding.PartitionSpec) -> bool:
  if len(lhs) < len(rhs):
    lhs, rhs = rhs, lhs
  if any(l is not None for l in lhs[len(rhs):]):
    return False
  return lhs[:len(rhs)] == rhs[:]


def check(dtype, shape_spec: ShapeSpec, value):
  """Checks that a value has the expected dtype and shape."""
  if not isinstance(value, jax.Array):
    raise TypeCheckError('is not a jax.Array')
  if value.dtype != dtype:
    raise TypeCheckError(f'is {value.dtype}, but expected {dtype}')
  shape = value.shape
  if len(shape) != len(shape_spec.dims):
    raise TypeCheckError(f'has shape {shape}, but expected shape {str(shape_spec)}')
  mesh = None

  axis_env = jax._src.core.thread_local_state.trace_state.axis_env
  if axis_env:
    # We're in a shard_map/pmap/xmap context. Multiply sizes by sharding, then check sizes.
    # We don't actually check the sharding, because that information is lost inside a 
    # shard_map/pmap/xmap context, but we do check the unsharded sizes are correct.
    mesh = {axis.name: axis.size for axis in axis_env}
    for orig_dim, dim_spec in zip(shape, shape_spec.dims):
      dim = orig_dim
      for axis in dim_spec.sharding:
        if axis not in mesh:
          raise TypeCheckError(f'has unknown mesh axis {axis}')
        axis_size = mesh[axis]
        dim *= axis_size
      check_size(dim_spec.shape, dim)
  else:
    # Check sizes
    for dim, dim_spec in zip(shape, shape_spec.dims):
      check_size(dim_spec.shape, dim)

    # Check sharding
    expected_spec = shape_spec.partition_spec()
    def cb(actual):
      if isinstance(actual, jax.sharding.SingleDeviceSharding):
        if any(dim_spec.sharding for dim_spec in shape_spec.dims):
          raise TypeCheckError(f'is fully replicated, but expected {expected_spec} is not')
      elif not isinstance(actual, jax.sharding.NamedSharding):
        if isinstance(actual, jax.sharding.Sharding):
          raise TypeCheckError(f'is SPMD-sharded but no axis names are available. Use `with Mesh(...):` to provide axis names for type checking.')
        else:
          raise TypeCheckError(f': unexpected object when checking sharding: {actual}')
      elif not _partition_spec_equiv(actual.spec, expected_spec):
        # TODO: when an axis size is None, recovering the NamedSharding from the PositionalSharding 
        # is ambiguous, and JAX often takes a different approach than the user does.
        #
        # We could fix this with a more precise _partition_spec_equiv, but for now we'll just ignore it.
        # raise TypeCheckError(f'has sharding spec {actual.spec}, but expected {expected_spec} from {str(shape_spec)}')
        pass
    # Use tracing as a proxy for whether we're in a jit context
    is_tracing = jax._src.core.thread_local_state.trace_state.trace_stack
    if is_tracing:
      jax.debug.inspect_array_sharding(value, callback=cb)
    else:
      cb(value.sharding)
    



#### Typeguard
def register_with_typeguard():
  """Registers the shardtypes module with typeguard. Call this at the beginning of your program."""
  def check_array(value, origin, args, memo):
    if len(args) != 1 or (type(args[0]) is not str and type(args[0]) is not bytes):
      raise TypeCheckError(f'has bad type signature; expected {origin.__name__}[<shape string>], got {origin.__name__}{args}')
    check(origin.dtype, ShapeSpec.parse(args[0]), value)

  def check_pytree_dataclass(value, origin, args, memo):
    if not isinstance(value, origin):
      raise TypeCheckError(f'is not an instance of {origin}')
    for field in dataclasses.fields(origin):
      check_type_internal(getattr(value, field.name), field.type, memo)

  def lookup(
      origin, args, extras
  ) -> TypeCheckerCallable | None:
    if isinstance(origin, type) and issubclass(origin, number):
      return check_array
    if origin in _PYTREE_DATACLASSES:
      return check_pytree_dataclass
    return None
  
  checker_lookup_functions.append(lookup)

#### Array types
class number:
  def __class_getitem__(cls, x):
    if isinstance(x, str):
      x = x.encode('utf-8')
    return GenericAlias(cls, x)

class bool_(number):
  dtype = jnp.bool_
  pass

class bf16(number):
  dtype = jnp.bfloat16
  pass

class f32(number):
  dtype = jnp.float32
  pass

class i32(number):
  dtype = jnp.int32
  pass

class u32(number):
  dtype = jnp.uint32
  pass

class i8(number):
  dtype = jnp.int8
  pass

class u8(number):
  dtype = jnp.uint8
  pass


_PYTREE_DATACLASSES = set()


def pytree_dataclass(cls):
  """Decorator that declares a dataclass that JAX recognizes as a PyTree."""
  cls = dataclass(cls)

  def flatten_with_keys(value):
    return [(k.name, getattr(value, k.name)) for k in dataclasses.fields(cls)], ()
  
  def unflatten(_aux, fields):
    return cls(*fields)

  jax.tree_util.register_pytree_with_keys(cls, flatten_with_keys, unflatten)
  _PYTREE_DATACLASSES.add(cls)
  return cls


def make_partition_specs(cls):
  """Instantiates a pytree dataclass with a PartitionSpec at array type."""
  # Check for a tuple type:
  origin = typing.get_origin(cls)
  args = typing.get_args(cls)
  if origin is tuple:
    return tuple(make_partition_specs(arg) for arg in args)
  elif origin is not None and issubclass(origin, number):
    if len(args) != 1 or (type(args[0]) is not str and type(args[0]) is not bytes):
      raise ValueError(f'Type annotation {cls} should be <dtype>[<shape string>], got {cls}')
    spec = ShapeSpec.parse(args[0])
    return spec.partition_spec()
  elif dataclasses.is_dataclass(cls):
    values = []
    for field in dataclasses.fields(cls):
      values.append(make_partition_specs(field.type))
    return cls(*values)
  
  raise ValueError(f'Unsupported type {cls} is not a array, dataclass, or tuple type')


def make_shardings(cls):
  """Instantiates a pytree dataclass with NamedSharding at array type."""
  mesh = jax._src.mesh.thread_resources.env.physical_mesh
  return jax.tree_map(lambda spec: jax.sharding.NamedSharding(mesh, spec), make_partition_specs(cls))


def typed_shard_map(f, **kwargs):
  """jax.shard_map, but which does not require specifying in_specs and out_specs.

  Instead, the function signature is used to infer the partitioning of the inputs and outputs.

  For example:
    @typed_shard_map
    def f(x: f32[b'batch/d len'], y: f32[b'e/d f/t']) -> f32[b'batch/d f/t']:
      ...
  
  """
  sig = inspect.signature(f)

  def wrapped(*args):
    mesh = jax._src.mesh.thread_resources.env.physical_mesh
    in_specs = tuple(make_partition_specs(param.annotation) for param in sig.parameters.values())
    out_specs = make_partition_specs(sig.return_annotation)
    return jax.experimental.shard_map.shard_map(typechecked(f), in_specs=in_specs, out_specs=out_specs, mesh=mesh, **kwargs)(*args)
  
  return wrapped

def is_fully_sharded(spec: jax.sharding.PartitionSpec):
  """Returns True if the spec is fully sharded, i.e. every device axis is used in the partition spec."""
  axis_count = 0
  for axis in spec:
    if axis is None:
      continue
    elif isinstance(axis, str):
      axis_count += 1
    elif isinstance(axis, tuple):
      axis_count += len(axis)
    else:
      raise ValueError(f'Unknown axis type {axis}')
  return axis_count == len(jax._src.core.thread_local_state.trace_state.axis_env)

def extend_named_axes(name: Union[bytes, str], cls):
  if isinstance(name, str):
    name = name.encode('utf-8')

  if dataclasses.is_dataclass(cls):
    extended_fields = []
    for fld in dataclasses.fields(cls):
      extended_type = extend_named_axes(name, fld.type)
      extended_fields.append((fld.name, extended_type))

    extended_cls = make_dataclass(cls.__name__, extended_fields, bases=(cls,))
    pytree_dataclass(extended_cls)
    return extended_cls
  else:
    number_type, shape = get_origin(cls), get_args(cls)
    extended_shape = (name + b' ' + shape[0],)
    return GenericAlias(number_type, extended_shape)

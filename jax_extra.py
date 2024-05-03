"""Extra utilities for JAX and Python."""
import jax
import hashlib
import jax
import jax.ad_checkpoint
import dataclasses
from typing import Union, get_args
from dataclasses import fields, is_dataclass


def fold_in_str(key: jax.Array, string: str) -> jax.Array:
  """Returns a PRNG key derived from an initial PRNG key and a string input.

  Args:
    key: The initial PRNG key.
    string: The string input (e.g., 'pretrain', 'query', etc.).

  Returns:
    A PRNG key derived from the initial PRNG key and the string input.
  """
  return jax.random.fold_in(
      key, int(hashlib.md5(string.encode()).hexdigest()[:8], base=16)
  )

def _convert(value, target_type):
    if value is None and target_type is not type(None):
      raise ValueError(f"Cannot convert None to {target_type}")
    elif value is None and target_type is type(None):
      return None
    elif is_dataclass(target_type):
        return make_dataclass_from_dict(target_type, value)
    else:
        return target_type(value)

def _handle_union(name, field_value, union_types):
    for type_option in union_types:
        try:
            return _convert(field_value, type_option)
        except (TypeError, ValueError, AssertionError):
            continue
    raise ValueError(f'could not convert Union type {name} to any of {union_types}.')

def make_dataclass_from_dict(cls, data):
    """Recursively instantiate a dataclass from a dictionary."""
    if data is None:
        raise ValueError(f'Expected a {cls.__name__}, got None instead.')
    field_data = {}
    for field in fields(cls):
        field_value = data.get(field.name)
        if hasattr(field.type, '__origin__') and field.type.__origin__ is Union:
            field_data[field.name] = _handle_union(field.name, field_value, get_args(field.type))
        else:
            try:
                field_data[field.name] = _convert(field_value, field.type)
            except (TypeError, ValueError, AssertionError):
                raise ValueError(f'Expected {field.type} for {cls.__name__}.{field.name}, got {type(field_value)} instead.')
    return cls(**field_data)

def explicit_activation_checkpointing(f):
  """Annotates a function f to be used with save_for_backward().
  
  Example:

  ```
  @explicit_activation_checkpointing
  def foo(W1, W2, W3, x):
    x = jax.nn.relu(save_for_backward(W1 @ x))
    x = jax.nn.relu(save_for_backward(W2 @ x))
    x = W3 @ x
  ```

  This causes the pre-ReLU activations to be saved for the backwards pass.
  """
  # We save everything that is named.
  return jax.ad_checkpoint.checkpoint(f, policy=jax.checkpoint_policies.save_any_names_but_these())

def save_for_backward(x):
  """Saves a value for the backwards pass in a function annotated with explicit_activation_checkpointing()."""
  # The actual name isn't important, just the fact that it _is_ named, so that 
  # the save_any_names_but_these() policy causes it to be saved.
  return jax.ad_checkpoint.checkpoint_name(x, name='seqax_save_for_backward')
"""Provides IO support for training:
* checkpoint save and load
* metrics logging
* profiling of XLA computations
* reporting FLOPs per device
"""
import jax
import jax.numpy as jnp
from jax.experimental import multihost_utils
from typing import Tuple, Any
from dataclasses import dataclass
import os
import fsspec
import zarr
from numcodecs import blosc
from clearml import Logger
import numpy as np
import datetime
import concurrent
import jax.profiler
import tempfile
import shutil
from jax.lib import xla_client

PyTree = Any

@dataclass
class IOConfig:
  # Max number of threads to use for IO-bound tasks like saving and loading checkpoints.
  # Recommendation: about 1MiB/thread is typical, so 1024 thread is reasonable for 1GiB of overhead.
  # Since this work is IO-bound rather than CPU-bound, it is fine to have many more threads than
  # CPU cores.
  max_io_threads: int

def log(step: int, logger: Logger, output: PyTree):
  """Logs the output of a training step. The output must be a PyTree of f32 arrays."""
  if jax.process_index() == 0:
    metrics_dict = {}
    for path, arr in jax.tree_util.tree_leaves_with_path(output):
      path = jax.tree_util.keystr(path)
      arr = jax.device_get(arr)
      if arr.shape == () and arr.dtype == jnp.float32:
        if logger:
          logger.report_scalar(
            title=path, series=path, value=arr, iteration=step)
        metrics_dict[path] = float(arr)
      elif arr.dtype == jnp.float32:
        if logger:
          logger.report_histogram(
            title=path, series=path, values=arr, iteration=step)
      else:
        raise ValueError(f"Output {path} has unsupported shape {arr.shape} and dtype {arr.dtype}.")
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{now}] Step {step}: {metrics_dict}")


def load_checkpoint_if_it_exists(checkpoint_dir: str, state: PyTree, config: IOConfig) -> Tuple[PyTree, int]:
  """Loads the latest checkpoint if it exists, otherwise return the initial state.

  In either case, uses the sharding and PyTree structure of `state` to produce the output.

  Since the state may occupy a large amount of memory, this function makes sure to delete `state`
  before loading the checkpoint. To facilitate this, callers should ensure not to hold on to any
  additional references to `state` when calling this function.

  Returns state and step number. Step 0 is the initial state, which may or may not have been loaded
  from a checkpoint.
  """
  blosc.use_threads = False  # Blindly following recommendation from https://zarr.readthedocs.io/en/stable/tutorial.html#parallel-computing-and-synchronization
  checkpoint_dir_pseudofile = fsspec.open(checkpoint_dir)
  fs = checkpoint_dir_pseudofile.fs
  checkpoint_dir_path = checkpoint_dir_pseudofile.path
  del checkpoint_dir_pseudofile

  # Check working_dir for checkpoint files.
  # Process index 0 selects the checkpoint, then broadcasts it to everyone else.
  selected_checkpoint = -1
  if jax.process_index() == 0:
    if fs.exists(checkpoint_dir_path):
      # fs.mkdir(checkpoint_dir, create_parents=False)
      checkpoint_dirs = fs.ls(checkpoint_dir_path)
      for c in reversed(sorted(checkpoint_dirs)):
        try:
          checkpoint_number = int(os.path.basename(c))
        except ValueError:
          continue
        root = zarr.open_group(zarr.storage.FSStore(c, fs=fs))
        if "write_completed" not in root.attrs:
          print(f"zarr 'write_completed' marker is missing in checkpoint {c}; skipping.")
          continue
        selected_checkpoint = checkpoint_number
        break
  selected_checkpoint = multihost_utils.broadcast_one_to_all(jnp.int32(selected_checkpoint))

  if selected_checkpoint == -1:
    print(f"No checkpoints found in {checkpoint_dir_path}, starting from initial state.")
    return state, 0
    
  print(f'Found checkpoint {selected_checkpoint} in {checkpoint_dir_path}, starting from there.')
  return load_zarr(os.path.join(checkpoint_dir, step_to_str(selected_checkpoint)), state, config), selected_checkpoint


def save_checkpoint(checkpoint_dir: str, step: int, state: PyTree, config: IOConfig):
  """Saves a checkpoint for the specified step number.
  
  See docs/pytree-zarr-checkpoint.md for the checkpoint format.
  """
  blosc.use_threads = False
  checkpoint_file = os.path.join(checkpoint_dir, step_to_str(step))
  if jax.process_index() == 0:
    # If there's already a checkpoint at this step, delete it. It might have been a partially
    # written checkpoint from a previous run.
    f = fsspec.open(checkpoint_dir)
    checkpoint_path = os.path.join(f.path, step_to_str(step))
    if f.fs.exists(checkpoint_path):
      f.fs.rm(checkpoint_path, recursive=True)
  
  print(f"[{datetime.datetime.now()}] Saving checkpoint {step} to {checkpoint_file}.")
  save_zarr(checkpoint_file, state, config)
  print(f"[{datetime.datetime.now()}] Finished saving checkpoint {step} to {checkpoint_file}.")


def load_zarr(filename: str, state: PyTree, config: IOConfig) -> PyTree:
  """Loads a zarr checkpoint from disk.
  
  See docs/pytree-zarr-checkpoint.md for the checkpoint format.
  """
  root = zarr.open_group(filename, mode="r")
  if "write_completed" not in root.attrs:
    raise ValueError(f"zarr 'write_completed' marker is missing. Should not have selected this checkpoint to load from.")

  def load_one(path: Tuple, prev: jax.Array) -> jax.Array:
    path = jax.tree_util.keystr(path)
    shape = prev.shape
    sharding = prev.sharding
    arr = root[path]
    assert arr.shape == shape, f'Expected shape {shape} but got {arr.shape} for {path} in {filename}'
    assert arr.dtype == prev.dtype, f'Expected dtype {prev.dtype} but got {arr.dtype} for {path} in {filename}'
    del prev  # Deallocate memory before loading its replacement!
    return jax.make_array_from_callback(shape, sharding, lambda shard_index: arr[shard_index])

  state, treedef = jax.tree_util.tree_flatten_with_path(state)
  with concurrent.futures.ThreadPoolExecutor(max_workers=config.max_io_threads) as executor:
    state_futures = [executor.submit(load_one, path, shape) for (path, shape) in state]
    states = [f.result() for f in state_futures]
  return jax.tree_util.tree_unflatten(treedef, states)


def save_zarr(filename: str, state: PyTree, config: IOConfig):
  """Saves a zarr checkpoint to disk.
  
  See docs/pytree-zarr-checkpoint.md for the checkpoint format.
  """
  state, _treedef = jax.tree_util.tree_flatten_with_path(state)

  if jax.process_index() == 0:
    # Create the zarr file and all the arrays.
    try:
      root = zarr.open_group(filename, mode='w-')
    except zarr.errors.ContainsGroupError:
      raise ValueError(f"Checkpoint {filename} already exists.")
    for path, arr in state:
      path = jax.tree_util.keystr(path)
      chunk_shape = arr.sharding.shard_shape(arr.shape)
      root.empty(path, shape=arr.shape, chunks=chunk_shape, dtype=arr.dtype)
  multihost_utils.sync_global_devices("save_zarr_begin")

  root = zarr.open_group(filename, mode='r+')

  def save_shard(dst: zarr.Array, shard: jax.Array, index: Tuple[int, ...]):
    dst[index] = np.asarray(shard)

  with concurrent.futures.ThreadPoolExecutor(max_workers=config.max_io_threads) as executor:
    for path, arr in state:
      path = jax.tree_util.keystr(path)
      dst = root[path]
      assert dst.chunks == arr.sharding.shard_shape(arr.shape)
      for shard in arr.addressable_shards:
        if shard.replica_id == 0:
          executor.submit(save_shard, dst, shard.data, shard.index)
  
  multihost_utils.sync_global_devices("save_zarr_end")
  if jax.process_index() == 0:
    root.attrs["write_completed"] = True
  multihost_utils.sync_global_devices("save_zarr_committed")

def step_to_str(step: int) -> str:
  """Converts a step number to a string with leading zeros.
  
  We pad up to 10 digits so that lexicographic order matches numerical. 1e10 training steps
  should be enough for anyone: the biggest runs as of 2024 are probably around 1e7 tokens/batch,
  1e13 tokens total, so 1e6 training steps total.
  """
  return str(step).zfill(10)

_PROFILE_DIR = None

def start_profile():
  """Starts gathering a JAX profile."""
  # Get fresh temporary directory
  global _PROFILE_DIR
  _PROFILE_DIR = tempfile.mkdtemp()
  print(f'[{datetime.datetime.now()}] Starting profile, saving to {_PROFILE_DIR}')
  jax.profiler.start_trace(_PROFILE_DIR, create_perfetto_trace=True)
  
def stop_profile(working_dir: str):
  """Stops gathering the JAX profile and saves it to a file."""
  global _PROFILE_DIR
  jax.profiler.stop_trace()
  print(f'[{datetime.datetime.now()}] Finished profile, copying to {working_dir}')
  fsspec_put(_PROFILE_DIR + '/', working_dir + '/')
  shutil.rmtree(_PROFILE_DIR)
  print(f'[{datetime.datetime.now()}] Finished copying profile to {working_dir}')
  _PROFILE_DIR = None


def fsspec_put(local_src: str, remote_dst: str):
  """Copies a file from local disk to a remote location specified by a fsspec path."""
  f = fsspec.open(remote_dst)
  fs = f.fs
  path = f.path
  del f
  print(f'Put {local_src} to {path}')
  fs.put(local_src, path, recursive=True, create_parents=True)


def save_hlo_svg(filespec: str, compiled: jax.stages.Compiled):
  """Saves a compiled function's HLO to an SVG file."""
  compiled_hlo_dot = xla_client._xla.hlo_module_to_dot_graph(compiled.runtime_executable().hlo_modules()[0])
  with tempfile.TemporaryDirectory() as d:
    with open(os.path.join(d, "hlo.dot"), "w") as f:
      f.write(compiled_hlo_dot)
    hlo_orig_svg = os.path.join(d, "hlo.original.svg")
    hlo_svg = os.path.join(d, "hlo.svg")
    os.system(f"dot -Tsvg {f.name} -o{hlo_orig_svg}")
    # Edit the SVG to remove everything before <svg>. There's a bunch of hover CSS that massively slows down
    # rendering in Chrome and adds little value: it just highlights edges when you hover over them.
    with open(hlo_orig_svg, "r") as f:
      svg = f.read()
      svg = svg[svg.index("<svg "):]
    with open(hlo_svg, "w") as f:
      f.write(svg)
    fsspec_put(hlo_svg, filespec)


def mkdir(filespec: str):
  """Creates a directory at the specified (possibly remote) fsspec path."""
  f = fsspec.open(filespec)
  fs = f.fs
  path = f.path
  del f
  if not fs.exists(path):
    fs.mkdir(path, create_parents=False)


def get_flops_per_device():
  """Gets the FLOPS per device for the current device kind."""
  device = jax.devices()[0].device_kind
  if device.startswith("NVIDIA A100"):
    result = 312e12
  else:
    print(f'Unrecognized device, assuming ridiculously low 1 MFLOPS. Device name: {device}')
    result = 1e6
  print(f'Device kind: {device}')
  print(f'FLOPS per device: {result:_}')
  return result
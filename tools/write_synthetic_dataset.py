# To run:
# 
# ```
# python write_synthetic_dataset.py --config-name=synthetic_dataset +output=synthetic_dataset.zarr
# ```
#
# Synthetic tasks: see Section 4 of https://arxiv.org/abs/2002.09402 for some ideas.
#
# We do:
# * Task 1: [theirs] fixed-distance copy. Requires attention position queries.
# * Task 2: [theirs] fixed-distance reverse. Requires attention position queries.
# * Task 3: [ours] random-distance (specified) copy. Requires variable-length position queries.
# * Task 4: [ours] random-distance (not specified) copy. Requires equality matching for a prefix.
# * Task 5: [ours] gaussians sampling. Requires model to learn which IDs are close to each other (in numerical order).
#
# Sequences begin with task ID, then have task-specific data. We avoid index 0, which indicates padding.

from functools import partial
import hydra
from jaxtyping import Float, Int, jaxtyped, UInt32
import numpy as np
from typeguard import typechecked as typechecker
from dataclasses import dataclass
from hydra.core.config_store import ConfigStore
import flat_tokens

@dataclass
class Config:
  output: str
  seed: int
  seq_len: int
  examples: int
  flat_tokens_config: flat_tokens.Config
  _target_: str = __name__ + '.Config'



@jaxtyped(typechecker=typechecker)
def copy(seq_len: int, examples: int, gen: np.random.Generator) -> UInt32[np.ndarray, 'batch seqlen']:
  seq = gen.integers(1, 11, (examples, (seq_len + 1) // 2), dtype=np.uint32)
  return np.append(seq, seq, axis=1)[:, :seq_len]

@jaxtyped(typechecker=typechecker)
def reverse(seq_len: int, examples: int, gen: np.random.Generator) -> UInt32[np.ndarray, 'batch seqlen']:
  seq = gen.integers(1, 11, (examples, (seq_len + 1) // 2), dtype=np.uint32)
  return np.append(seq, np.flip(seq, axis=1), axis=1)[:, :seq_len]

@jaxtyped(typechecker=typechecker)
def random_known_distance_copy(seq_len: int, examples: int, gen: np.random.Generator) -> UInt32[np.ndarray, 'batch seqlen']:
  distance = gen.integers(max(1, seq_len // 4), seq_len, (examples,), dtype=np.uint32)
  seq = gen.integers(1, 11, (examples, seq_len), dtype=np.uint32)
  indices = np.arange(seq_len - 1)[np.newaxis, :] % distance[:, np.newaxis]
  full_seq = seq[np.arange(examples)[:, np.newaxis], indices]
  assert full_seq.shape == (examples, seq_len - 1)
  return np.append(distance[:, np.newaxis], full_seq, axis=1)

@jaxtyped(typechecker=typechecker)
def random_unknown_distance_copy(seq_len: int, examples: int, gen: np.random.Generator) -> UInt32[np.ndarray, 'batch seqlen']:
  return random_known_distance_copy(seq_len + 1, examples, gen)[:, 1:]

@jaxtyped(typechecker=typechecker)
def mixture_of_gaussians(seq_len: int, examples: int, gen: np.random.Generator) -> UInt32[np.ndarray, 'batch seqlen']:
  centers = gen.uniform(0, 100, (examples, 3)).astype(np.float32)
  stddevs = gen.uniform(1, 4, (examples, 3)).astype(np.float32)
  sample_cluster_ids = gen.integers(0, 3, (examples, seq_len), dtype=np.uint32)
  batch_ids = np.arange(examples)[:, np.newaxis]
  sample_centers = centers[batch_ids, sample_cluster_ids]
  sample_stddevs = stddevs[batch_ids, sample_cluster_ids]
  floats = gen.normal(0, 1, (examples, seq_len)).astype(np.float32) * sample_stddevs + sample_centers
  return np.clip(np.round(floats).astype(np.uint32), 1, 100)

@jaxtyped(typechecker=typechecker)
def synthetic_task(config: Config, gen: np.random.Generator) -> list[UInt32[np.ndarray, '...']]:
  task_seq_len = config.seq_len - 1
  examples = config.examples
  copy_data = copy(task_seq_len, examples, gen)
  reverse_data = reverse(task_seq_len, examples, gen)
  random_known_distance_copy_data = random_known_distance_copy(task_seq_len, examples, gen)
  random_unknown_distance_copy_data = random_unknown_distance_copy(task_seq_len, examples, gen)
  mixture_of_gaussians_data = mixture_of_gaussians(task_seq_len, examples, gen)
  tasks = np.asarray([copy_data, reverse_data, random_known_distance_copy_data, random_unknown_distance_copy_data, mixture_of_gaussians_data])
  task_id = gen.integers(1, 6, (examples,), dtype=np.uint32)
  targets = np.append(task_id[:, np.newaxis], tasks[task_id - 1, np.arange(examples)], axis=1)
  lengths = gen.integers(1, config.seq_len + 1, (examples,), dtype=np.uint32)
  ragged_targets = [targets[i, :lengths[i]] for i in range(examples)]
  return ragged_targets


# Registering the Config class with the name 'config'.
ConfigStore.instance().store(name="config_schema", node=Config)


@hydra.main(config_path="configs", version_base=None)
def main(config):
  config = hydra.utils.instantiate(config)
  gen = np.random.Generator(np.random.PCG64(config.seed))

  for split, mode in [(flat_tokens.Split.VALIDATION, "w-"), (flat_tokens.Split.TRAIN, "r+")]:
    dst = flat_tokens.Writer(config.output, split, mode, config.flat_tokens_config)
    examples = synthetic_task(config, gen)
    dst.write(flat_tokens.Chunk.from_ragged(examples))

if __name__ == "__main__":
  main()
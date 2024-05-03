# seqax = sequence modeling + JAX

seqax is a codebase for small-to-medium-scale LLM pretraining research. The entire training program---including the model implementation; optimizer; multihost FSDP and tensor parallel partitioning---is [500 lines of code](/train.py), which scales well up to ~100 GPUs or TPUs[^1] and [typically achieves good MFUs of 30-50%](#performance).

[^1]: Achieving good performance at larger scale requires pipeline parallelism (which we have not yet implemented). At that scale, you may also care more about using custom kernels to further improve performance at the cost of code simplicity.

seqax is written in a style that makes the important information visible, rather than being hidden behind abstractions and indirections or being inferred automatically and unpredictably. This shows up in:

* **Math**. seqax implements all of the training step's math, rather than calling into external libraries. If you want to understand or change the math, it's right there!

* **Memory**. All tensors that go into a model checkpoint on disk are explicits. All tensors that occupy a lot of memory, including activations saved for the backwards pass, are explicit. You can straightforwardly read the memory footprint from the source code.

* **Partitioning and communication**. The partitioned layout of all tensors and operations is explicit. All interchip communication is explicit.

## Getting started

### Installation

1. Install `graphviz` from your system package manager: e.g. `brew install graphviz` or `apt install graphviz`.
2. Install Python dependencies, typically inside a virtualenv: `python -m pip install -r requirements.txt`

### Run on CPU for local development

For development and testing you can run on CPU. Typically you'd use our synthetic dataset (which is [checked into this repository](/synthetic_dataset.zarr)) or the [Huggingface data loader](#data-loaders) and you'd set XLA flags to simulate multiple devices so as to test that parallelism is working as intended:

```
XLA_FLAGS=--xla_force_host_platform_device_count=8 python -m train --config-name=local_test_synthetic +paths.model_name=synthetic_000
```

The `paths.model_name` flag specifies which subdirectory on disk (inside `/tmp`) to write model checkpoints to. You'll typically want to change this when starting a new model run.

### Run on GPUs

We have configured a range of model sizes, to be trained on the C4 dataset with the Llama tokenizer. Browse the `configs/` directory to select your preferred configuration file. Each configuration file lists how to run it at the top.

You typically want to set `paths.model_name` to a unique name for each distinct training run. This path specifies which subdirectory on disk to write model checkpoints to.

## Performance

Recent benchmark results on A100 clusters:

Single-host A100x8
| Model Size | MFU   |
|------------|-------|
| 84m        | 14    |
| 270m       | 24    |
| 540m       | 35    |
| 1b         | 41.6  |
| 2b         | 50.66 |

On 4 A100x8 hosts connected with infiniband
| Model Size | MFU   |
|------------|-------|
| 1b         | 32.4  |
| 2b         | 39.0  |

## Data loaders

seqax can stream training data directly from Huggingface (see [example config](/configs/huggingface_c4_a100x1_84m.yaml)), or can first convert the training data to a pre-tokenized format on disk which we call [flat-tokens](/docs/flat-tokens.md) (see [example config](/configs/flat_tokens_c4_a100x1_84m.yaml)). Streaming from Huggingface allows you to quickly experiment with different datasets, but it doesn't offer an efficient way to resume training from a checkpoint after a job is aborted, and it wastes some tokens from the dataset at batch boundaries. The flat-tokens format supports efficiently resuming training from a checkpoint, uses 100% of tokens for training, and also consumes less CPU time during training.

To pre-tokenize the training data, you can run [huggingface_to_flat_tokens.py](/tools/huggingface_to_flat_tokens.py). You'll need to first install the requirements in [/tools/requirements.txt](/tools/requirements.txt), and then you can invoke the command listed at the top of [/tools/configs/c4_en.yaml](/tools/configs/c4_en.yaml). On modern CPUs this script processes about 100M tokens per minute. You can limit the number of output tokens it processes with a configuration flag.

## Expressing partitioning and communication with `shardlib`

seqax ships with a new library called [shardlib](/shardlib) for expressing partitioning and communication with JAX, building on the ideas and style of [jaxtyping](https://docs.kidger.site/jaxtyping/), [einops](https://einops.rocks/), [equinox](https://docs.kidger.site/equinox/), and [shard_map](https://jax.readthedocs.io/en/latest/jep/14273-shard-map.html). Here we demonstrate its core ideas, to implement fully sharded data parallelism (FSDP) for a simple fully connected neural network.


```python
# XLA_FLAGS=--xla_force_host_platform_device_count=8 python -m shardlib_example
from shardlib.shardtypes import bf16, bool_, f32, pytree_dataclass, typed_shard_map, u32, make_shardings
from shardlib import shardtypes
shardtypes.register_with_typeguard()
import shardlib.shardops as shardops
from jax.sharding import Mesh
from jax.experimental import mesh_utils
import jax
import jax.numpy as jnp

# We set up a device mesh where 'd' refers to the "data parallel" axis.
MESH = Mesh(mesh_utils.create_device_mesh([8], jax.devices()), ('d'))

# At rest, weights are all sharded over the data parallel axis, making them fully sharded.
#
# The `hidden1/d` syntax means that second axis has size `hidden1` and is sharded over device axis `d`.
# Equivalently, you can view this as saying that the per-device shape is `(in, hidden1/d)`, where `/`
# indicates division.
@pytree_dataclass
class Weights:
  w1: f32['in hidden1/d']
  w2: f32['hidden1 hidden2/d']
  w3: f32['hidden2/d']

with MESH:
  # Create dummy weights.
  w = Weights(
    w1=jnp.zeros((8, 8), dtype=jnp.float32),
    w2=jnp.zeros((8, 8), dtype=jnp.float32),
    w3=jnp.zeros((8,), dtype=jnp.float32),
  )
  # Apply sharding to weights. The sharding specs are inferred from the type annotations on the Weights class.
  w = jax.tree.map(jax.device_put, w, make_shardings(Weights))

  # We use `typed_shard_map` to allow us to write per-device code with explicit communication.
  #
  # Compared to untyped `jax.shard_map`, the `in_specs` and `out_specs` do not need to be specified:
  # they're inferred from the sharding on the function's signature.
  @typed_shard_map
  def forward_pass(x: f32[b'batch/d in'], w: Weights) -> f32[b'batch/d']:
    # Weights are all-gathered just prior to their use. (This is the core idea of fully-sharded data parallelism.)
    # The `in hidden1/d -> in hidden1` syntax expresses what this all-gather operation should do: it removes the
    # `d` sharding on the `hidden1` axis, resulting in a fully replicated output.
    w1 = shardops.all_gather('in hidden1/d -> in hidden1', w.w1)
    # The `einsum_unreduced` operation is a chip-local einsum. Unlike `jnp.einsum`, it supports sharding syntax,
    # and it performs shape checking using the current typing environment, so it will raise an error if for example
    # you use `batch` in two different ways within a function.
    #
    # We call this einsum "unreduced", because it does not do any cross-chip reductions, even if they are necessary.
    # For example, in an `a b/d, b/d c -> a c` einsum, a cross-chip reduction over the `d` sharding axis is required,
    # and it is the caller's responsibility to perform this reduction.
    y = jax.nn.relu(shardops.einsum_unreduced('batch/d in, in hidden1 -> batch/d hidden1', x, w1))
    w2 = shardops.all_gather('hidden1 hidden2/d -> hidden1 hidden2', w.w2)
    z = jax.nn.relu(shardops.einsum_unreduced('batch/d hidden1, hidden1 hidden2 -> batch/d hidden2', y, w2))
    w3 = shardops.all_gather('hidden2/d -> hidden2', w.w3)
    return shardops.einsum_unreduced('batch/d hidden2, hidden2 -> batch/d', z, w3)

  x = forward_pass(jnp.zeros((32, 8), dtype=jnp.float32), w)
  assert(x.shape == (32,))
```

There are several other APIs exported by shardlib in addition to the ones demonstrated here. [Browse the code](/shardlib/) to see the full list.

## Expressing activation checkpointing using `save_for_backward`

Which intermediate computations in the forwards pass are saved to HBM for later use in the backwards pass? [The default answer](https://jax.readthedocs.io/en/latest/notebooks/autodiff_remat.html) is: JAX saves _all_ intermediates for use in the backwards pass, but in JIT mode the XLA compiler optimizes many of these away so as to save memory.

While JAX provides many sophisticated policies for making these choices, we offer a very simple one: calling `save_for_backward` causes its argument to be saved for the backwards pass. Here is an example:

```python
from jax_extra import explicit_activation_checkpointing, save_for_backward

# The @explicit_activation_checkpointing switches JAX from its default
# policy of saving all intermediates, and instead only saves the
# arguments to the annotated function, plus any intermediates marked
# with `save_for_backward`.
@explicit_activation_checkpointing
def forward_pass(x, w1, w2):
  # save_for_backward marks `y` as being saved.
  y = save_for_backward(x @ w1)
  # `z` is not saved for the backwards pass.
  z = jax.nn.relu(z)
  return z @ w2
```

## Profiling

Every training run gathers and reports performance information:
* the time for two training steps (including data fetching in between them). This is written to stdout.
* model FLOPS utilization (MFU) efficiency for these steps. This is written to stdout.
* an XLA performance profile. This is written into the model directory at `<model_dir>/plugins/profile/<date>/perfetto_trace.json.gz`
* an rendered SVG of the optimized XLA computation graph. This is written into the model directory at `<model_dir>/training_step_optimized_hlo_<date>.svg`.


## File formats

We write checkpoints and datasets in simple file formats based on [zarr](https://zarr.dev/). See our file format specifications:
* [our checkpoint format](/docs/pytree-zarr-checkpoint.md)
* [our dataset format](/docs/flat-tokens.md)

## Acknowledgements

seqax's implementation style was substantially inspired by [jaxtyping](https://docs.kidger.site/jaxtyping/), [einops](https://einops.rocks/), [equinox](https://docs.kidger.site/equinox/), and [shard_map](https://jax.readthedocs.io/en/latest/jep/14273-shard-map.html).

Thanks to [MaxText](https://github.com/google/maxtext) for demonstrating good practices for production LLM use of JAX.

Thanks to the [JAX](https://github.com/google/jax) team for ongoing support and advice.

Thanks to the [Google TPU Research Cloud](https://sites.research.google/trc/about/), which partially supported this work.

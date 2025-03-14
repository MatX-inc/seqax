"""Main training loop, including the model, loss function, and optimizer."""
import operator
import os
import time

import env
env.set_variables()
import shardlib.shardtypes as shardtypes
shardtypes.register_with_typeguard()
import gcsfs  # Needed for clearml setup

import datetime
from functools import cached_property, partial
from typing import Any, Optional, Tuple, Union
import hydra
from typeguard import typechecked
from dataclasses import dataclass
import jax
from jax import lax
from jax.sharding import PartitionSpec
import jax.numpy as jnp
import math
from input_loader import FlatTokensParams, HuggingFaceDataParams, TokenBatch, TokenBatchParams, get_loader
from shardlib.shardtypes import bf16, bool_, f32, pytree_dataclass, u32, make_shardings, Array
import shardlib.shardops as shardops
P = PartitionSpec
import einops
import jax_extra
from jax_extra import fold_in_str, explicit_activation_checkpointing, save_for_backward
import os
import training_io
from clearml import Task
from jax.experimental import mesh_utils
from jax.sharding import Mesh
from jax.tree_util import tree_leaves

PRNGKey = Any

@dataclass(frozen=True)
class Hparams:
  d_model: int
  n_q_per_kv: int
  n_kv: int
  d_head: int
  layers: int
  vocab: int
  d_ff: int
  rope_max_timescale: int

@pytree_dataclass
class TransformerLayer:
  ln1: f32['d_model/t/d']
  ln2: f32['d_model/t/d']
  w_q: f32['d_model/d n_q_per_kv n_kv/t d_head']
  w_kv: f32['2 d_model/d n_kv/t d_head']
  w_o: f32['d_model/d n_q_per_kv n_kv/t d_head']
  w_gate: f32['d_model/d d_ff/t']
  w_up: f32['d_model/d d_ff/t']
  w_down: f32['d_model/d d_ff/t']

Transformer = Array['layers', TransformerLayer]

@pytree_dataclass
class Model:
  embed: f32['vocab/t d_model/d']
  unembed: f32['vocab/t d_model/d']
  transformer: Transformer
  final_layer_norm: f32['d_model/d/t']

  @staticmethod
  @typechecked
  def init(h: Hparams, rng: PRNGKey) -> 'Model':
    embed = jax.random.normal(jax_extra.fold_in_str(rng, 'embed'), (h.vocab, h.d_model), dtype=jnp.float32)
    # https://github.com/google/jax/issues/20390 for ones_like with sharding.
    ln1 = jnp.ones((h.layers, h.d_model), dtype=jnp.float32)
    ln2 = jnp.ones((h.layers, h.d_model), dtype=jnp.float32)
    final_layer_norm = jnp.ones((h.d_model,), dtype=jnp.float32)
    
    # All of wi/wq/wo/wo/w_kv use truncated_normal initializers with 'fan_in' scaling,
    # i.e. variance set to 1.0/fan_in.
    # The constant is stddev of standard normal truncated to (-2, 2)
    truncated_normal_stddev = .87962566103423978

    # scale for tensors with d_model fan_in and truncated normal truncated to (-2, 2)
    d_model_scale = 1 / (math.sqrt(h.d_model) * truncated_normal_stddev)

    w_kv_scale = d_model_scale
    w_q_scale = d_model_scale / math.sqrt(h.d_head)
    total_head_dim = h.n_q_per_kv * h.n_kv * h.d_head
    w_o_scale = 1 / (math.sqrt(total_head_dim) * truncated_normal_stddev)
    w_up_scale = d_model_scale
    w_down_scale = 1 / (math.sqrt(h.d_ff) * truncated_normal_stddev)
    unembed_scale = d_model_scale

    w_q_shape = (h.layers, h.d_model, h.n_q_per_kv, h.n_kv, h.d_head)
    w_q = w_q_scale * jax.random.truncated_normal(fold_in_str(rng, 'w_q'), -2, 2, w_q_shape, dtype=jnp.float32)
    w_kv_shape = (h.layers, 2, h.d_model, h.n_kv, h.d_head)
    w_kv = w_kv_scale * jax.random.truncated_normal(fold_in_str(rng, 'w_kv'), -2, 2, w_kv_shape, dtype=jnp.float32)
    w_o_shape = w_q_shape
    w_o = w_o_scale * jax.random.truncated_normal(fold_in_str(rng, 'w_o'), -2, 2, w_o_shape, dtype=jnp.float32)

    ff_shape = (h.layers, h.d_model, h.d_ff)
    w_gate = w_up_scale * jax.random.truncated_normal(fold_in_str(rng, 'w_gate'), -2, 2, ff_shape, dtype=jnp.float32)
    w_up = w_up_scale * jax.random.truncated_normal(fold_in_str(rng, 'w_up'), -2, 2, ff_shape, dtype=jnp.float32)
    w_down = w_down_scale * jax.random.truncated_normal(fold_in_str(rng, 'w_down'), -2, 2, ff_shape, dtype=jnp.float32)

    unembed = unembed_scale * jax.random.truncated_normal(fold_in_str(rng, 'unembed'), -2, 2, (h.vocab, h.d_model), dtype=jnp.float32)
    arrays = Model(
      embed=embed,
      unembed=unembed,
      transformer=Transformer(
        ln1=ln1,
        ln2=ln2,
        w_q=w_q,
        w_kv=w_kv,
        w_o=w_o,
        w_gate=w_gate,
        w_up=w_up,
        w_down=w_down,
      ),
      final_layer_norm=final_layer_norm,
    )
    shardings = make_shardings(Model)
    return jax.tree.map(lax.with_sharding_constraint, arrays, shardings)


  @typechecked
  def forward_pass(self, h: Hparams, ids: u32[b'B/d L'], is_seq_start: bool_[b'B/d L']) -> f32[b'B/d L V/t']:
    ##### Initial embedding lookup.
    embed = shardops.all_gather('V/t M/d -> V/t M', jnp.bfloat16(self.embed))
    x = shardops.index_unreduced('[V/t] M, B/d L -> B/d L M', embed, ids)
    x = shardops.psum_scatter('B/d L M -> B/d L M/t', x)

    L = ids.shape[1]
    segment_ids = jnp.cumsum(is_seq_start, axis=1)
    segment_mask: bool_[b'B/d L L'] = segment_ids[:, :, jnp.newaxis] == segment_ids[:, jnp.newaxis, :]
    segment_mask: bool_[b'B/d L L 1 1'] = segment_mask[..., jnp.newaxis, jnp.newaxis] # add axes for q_per_k, num_kv_heads dimensions
    causal_mask: bool_[b'1 L L 1 1'] = jnp.tril(jnp.ones((L, L), dtype=jnp.bool_), 0)[jnp.newaxis, ..., jnp.newaxis, jnp.newaxis]
    causal_mask: bool_[b'B/d L L 1 1'] = jnp.logical_and(segment_mask, causal_mask)

    rope_table = RopeTable.create(L, h)

    ##### Transformer blocks.
    @explicit_activation_checkpointing
    @typechecked
    def loop_body(x: bf16[b'B/d L M/t'], layer_weights: TransformerLayer) -> Tuple[bf16[b'B/d L M/t'], Tuple[()]]:
      # Pre-attention RMSNorm
      ln1 = shardops.all_gather('M/t/d -> M', jnp.float32(layer_weights.ln1))
      gx = shardops.all_gather('B/d L M/t -> B/d L M', x)
      nx = jnp.bfloat16(rms_norm(gx) * ln1)

      # Attention, using Grouped Query Attention and RoPE position embeddings.
      w_q = shardops.all_gather('M/d Q K/t D -> M Q K/t D', jnp.bfloat16(layer_weights.w_q))
      q = save_for_backward(shardops.einsum_unreduced('B/d L M, M Q K/t D -> B/d L Q K/t D', nx, w_q))
      q = rope_table.apply('L D -> 1 L 1 1 D', q)
      w_kv = shardops.all_gather('2 M/d K/t D -> 2 M K/t D', jnp.bfloat16(layer_weights.w_kv))
      k, v = shardops.einsum_unreduced('B/d L M, k_v M K/t D -> k_v B/d L K/t D', nx, w_kv)
      k = save_for_backward(k)
      v = save_for_backward(v)
      k = rope_table.apply('L d -> 1 L 1 d', k)
      logits = shardops.einsum_unreduced(
        'B/d Qlen Q K/t D, B/d Klen K/t D -> B/d Qlen Klen Q K/t', q, k, preferred_element_type=jnp.float32)
      logits = jnp.where(causal_mask, logits, -1e10)
      probs = jnp.bfloat16(jax.nn.softmax(logits, axis=2))
      attn_out = shardops.einsum_unreduced(
        'B/d Qlen Klen Q K/t, B/d Klen K/t D -> B/d Qlen Q K/t D', probs, v)
      w_o = shardops.all_gather('M/d Q K/t D -> M Q K/t D', jnp.bfloat16(layer_weights.w_o))
      attn_out = shardops.einsum_unreduced('B/d Qlen Q K/t D, M Q K/t D -> B/d Qlen M', attn_out, w_o)
      attn_out = shardops.psum_scatter('B/d Qlen M -> B/d Qlen M/t', attn_out)
      x = save_for_backward(x + attn_out)

      # Pre-FFN RMSNorm
      ln2 = save_for_backward(shardops.all_gather('M/t/d -> M', jnp.float32(layer_weights.ln2)))
      gx = shardops.all_gather('B/d L M/t -> B/d L M', x)
      nx = jnp.bfloat16(rms_norm(gx) * ln2)

      # FFN, using SwiGLU
      w_gate = shardops.all_gather('M/d F/t -> M F/t', jnp.bfloat16(layer_weights.w_gate))
      gate_proj = save_for_backward(shardops.einsum_unreduced('B/d L M, M F/t -> B/d L F/t', nx, w_gate))
      w_up = shardops.all_gather('M/d F/t -> M F/t', jnp.bfloat16(layer_weights.w_up))
      up_proj = save_for_backward(shardops.einsum_unreduced('B/d L M, M F/t -> B/d L F/t', nx, w_up))
      y = jax.nn.swish(gate_proj) * up_proj
      w_down = shardops.all_gather('M/d F/t -> M F/t', jnp.bfloat16(layer_weights.w_down))
      ffn_out = shardops.einsum_unreduced('B/d L F/t, M F/t -> B/d L M', y, w_down)
      ffn_out = shardops.psum_scatter('B/d L M -> B/d L M/t', ffn_out)

      return jnp.bfloat16(x + ffn_out), ()

    x, () = jax.lax.scan(loop_body, jnp.bfloat16(x), self.transformer)

    ##### Final layernorm and output projection.
    x = shardops.all_gather('B/d L M/t -> B/d L M', x)
    ln = shardops.all_gather('M/t/d -> M', jnp.float32(self.final_layer_norm))
    x = jnp.bfloat16(rms_norm(x) * ln)
    unembed = shardops.all_gather('V/t M/d -> V/t M', jnp.bfloat16(self.unembed))
    logits = shardops.einsum_unreduced('B/d L M, V/t M -> B/d L V/t', x, unembed, preferred_element_type=jnp.float32)

    return logits


  @typechecked
  def loss(self, h: Hparams, batch: TokenBatch) -> f32[b'']:
    # Given sequence-packed targets:
    #   [[1, 2], [3, 4, 5], [6, 7, 8, 9]]
    # we want inputs:
    #   [[0, 1], [0, 3, 4], [0, 6, 7, 8]]
    # which we get by shifting the targets right by 1 and 
    # masking sequence-start tokens to 0.
    inputs = jnp.pad(batch.targets[:, :-1], pad_width=((0, 0), (1, 0)))
    is_seq_start: bool_[b'batch/d len'] = batch.is_seq_start
    inputs: u32[b'batch/d len'] = jnp.where(is_seq_start, 0, inputs)

    logits: f32[b'batch/d len V/t'] = self.forward_pass(h, inputs, is_seq_start)
    max_logits: f32[b'batch/d len 1'] = lax.pmax(jnp.max(lax.stop_gradient(logits), axis=-1, keepdims=True), 't')
    logits = logits - max_logits
    sum_logits = lax.psum(jnp.sum(jnp.exp(logits), axis=-1, keepdims=True), 't')
    logsumexp = jnp.log(sum_logits)
    logprobs: f32[b'batch/d len V/t'] = logits - logsumexp
    logprobs_at_targets = shardops.index_unreduced('batch/d len [V/t], batch/d len -> batch/d len', logprobs, batch.targets)
    logprobs_at_targets = shardops.psum_scatter('batch/d len -> batch/d len/t', logprobs_at_targets)
    tokens_in_global_batch = logprobs_at_targets.size * jax.lax.psum(1, ('d', 't'))
    return -jnp.sum(logprobs_at_targets) / jnp.float32(tokens_in_global_batch)


@pytree_dataclass
class RopeTable:
  sin: f32['len d_head2']
  cos: f32['len d_head2']

  @staticmethod
  def create(max_len: int, hparams: Hparams) -> 'RopeTable':
    rope_max_timescale = hparams.rope_max_timescale
    d_head = hparams.d_head
    d = d_head // 2
    # endpoint=False is equivalent to what MaxText does. endpoint=True would be more natural, though.
    timescale = jnp.logspace(0, jnp.log10(jnp.float32(rope_max_timescale)), d, endpoint=False)
    position = jnp.arange(max_len, dtype=jnp.int32)
    sinusoid_inp = jnp.float32(position[:, jnp.newaxis]) / timescale[jnp.newaxis, :]
    sin = jnp.sin(sinusoid_inp)
    cos = jnp.cos(sinusoid_inp)
    return RopeTable(sin=sin, cos=cos)
  
  def apply(self, rearrange_spec, x):
    x1, x2 = jnp.split(x, 2, axis=-1)
    sin = einops.rearrange(self.sin, rearrange_spec)
    cos = einops.rearrange(self.cos, rearrange_spec)
    r1 = x1 * cos - x2 * sin
    r2 = x2 * cos + x1 * sin
    return jnp.append(r1, r2, axis=-1)


@typechecked
def rms_norm(x: bf16[b'batch/d len M']) -> bf16[b'batch/d len M']:
  mean2 = save_for_backward(jnp.mean(jax.lax.square(jnp.float32(x)), axis=-1, keepdims=True))
  return jnp.bfloat16(x * jax.lax.rsqrt(mean2 + 1e-6))


@pytree_dataclass
class Metrics:
  loss: f32[b'']
  learning_rate: f32[b'']
  grad_norm: f32[b'']
  raw_grad_norm: f32[b'']


@dataclass(frozen=True)
class TrainingHparams:
  adam_b1: float
  adam_b2: float
  adam_eps: float
  adam_eps_root: float
  weight_decay: float
  warmup_steps: int
  steps: int
  steps_for_lr: int
  cosine_learning_rate_final_fraction: float
  learning_rate: float
  tokens: TokenBatchParams
  seed: int
  queue: Optional[str] = None

@pytree_dataclass
class State:
  weights: Model
  adam_mu: Model
  adam_nu: Model

  @staticmethod
  def init(hparams: Hparams, rng: PRNGKey) -> 'State':
    weights = Model.init(hparams, rng)
    adam_mu = jax.tree.map(lambda p: p * 0.0, weights)
    adam_nu = jax.tree.map(lambda p: p * 0.0, weights)
    return State(weights=weights, adam_mu=adam_mu, adam_nu=adam_nu)

@partial(jax.jit, static_argnums=(2, 3), donate_argnums=(0,))
def training_step(state: State, step: u32[b''], h: Hparams, hparams: TrainingHparams, batch: TokenBatch) -> Tuple[Any, Metrics]:
  @partial(shardtypes.typed_shard_map, check_rep=False)  # check_rep=False for https://github.com/google/jax/issues/20335
  def sharded_step(state: State, step: u32[b''], batch: TokenBatch) -> Tuple[State, Metrics]:
    loss, grad = jax.value_and_grad(lambda weights: weights.loss(h, batch))(state.weights)
    # Gradients have already been reduced across chips because the gradient of the weight `all_gather`
    # is weight-gradient `psum_scatter`. Loss, on the other hand, hasn't been reduced across chips: if we
    # did that inside the autodiff, we'd be double-reducing the loss, effectively multiplying it by the 
    # amount of data parallelism.
    #
    # So we reduce the loss across chips _outside_ the autodiff.
    loss = jax.lax.psum(loss, ('d', 't'))

    # Other than global-norm of gradients, no other communication is needed during the weight update,
    # because weights and grads are already fully sharded, as checked below.

    # Calculate learning rate from step number.
    # We use linear warmup then cosine decay. See https://arxiv.org/pdf/2307.09288.pdf section 2.2
    warmup_lr = (jnp.float32(step) / jnp.float32(hparams.warmup_steps)) * hparams.learning_rate
    cosine = jnp.cos(jnp.pi * (jnp.float32(step - hparams.warmup_steps) / jnp.float32(hparams.steps_for_lr - hparams.warmup_steps)))
    cosine_lr = hparams.learning_rate * (hparams.cosine_learning_rate_final_fraction + (1 - hparams.cosine_learning_rate_final_fraction) * (cosine * .5 + .5))
    lr = jnp.where(step < hparams.warmup_steps, warmup_lr, cosine_lr)    

    # AdamW optimizer with global gradient clipping.
    grad_leaves, grad_treedef = jax.tree_util.tree_flatten(grad)
    global_norm_square = jnp.float32(0.0)
    for g in grad_leaves:
      assert g.dtype == jnp.float32
      global_norm_square += jnp.sum(jax.lax.square(g))
    global_norm_square = jax.lax.psum(global_norm_square, ('d', 't'))
    global_norm = jnp.sqrt(global_norm_square)
    rescale = jnp.minimum(1.0, 1.0 / global_norm)

    new_ps = []
    new_mus = []
    new_nus = []
    for p, g, mu, nu, spec in zip(tree_leaves(state.weights), grad_leaves, tree_leaves(state.adam_mu), tree_leaves(state.adam_nu), tree_leaves(shardtypes.make_partition_specs(State))):
      assert shardtypes.is_fully_sharded(spec), 'Weight update is only correctly scaled for fully sharded weights.'
      # Gradient clipping
      g = g * rescale
      # Adam scaling
      mu = (1 - hparams.adam_b1) * g + hparams.adam_b1 * mu
      nu = (1 - hparams.adam_b2) * jax.lax.square(g) + hparams.adam_b2 * nu
      # We need step numbers to start at 1, not 0. Otherwise the bias correction produces NaN.
      completed_steps = step + 1  
      mu_hat = mu / (1 - jnp.float32(hparams.adam_b1)**completed_steps)
      nu_hat = nu / (1 - jnp.float32(hparams.adam_b2)**completed_steps)
      g = mu_hat / (jnp.sqrt(nu_hat + hparams.adam_eps_root) + hparams.adam_eps)
      # Weight decay
      g += hparams.weight_decay * p
      # Learning rate
      g *= lr

      # Apply update
      new_ps.append(p - g)
      new_mus.append(mu)
      new_nus.append(nu)
    
    new_state = State(
      weights=jax.tree_util.tree_unflatten(grad_treedef, new_ps),
      adam_mu=jax.tree_util.tree_unflatten(grad_treedef, new_mus),
      adam_nu=jax.tree_util.tree_unflatten(grad_treedef, new_nus),
    )
    metrics = Metrics(
      loss=loss,
      learning_rate=lr,
      grad_norm=global_norm * rescale,
      raw_grad_norm=global_norm,
    )
    return new_state, metrics
  
  return sharded_step(state, step, batch)


@dataclass(frozen=True)
class Paths:
  root_working_dir: str
  model_name: str

@dataclass(frozen=True)
class MeshConfig:
  d: int
  t: int


@dataclass(frozen=True)
class Config:
  model: Hparams
  training: TrainingHparams
  paths: Paths
  num_hosts: int
  checkpoint_interval: int
  mesh: MeshConfig
  io: training_io.IOConfig
  flat_tokens: Optional[FlatTokensParams] = None
  hf_dataset: Optional[HuggingFaceDataParams] = None

  def __post_init__(self):
    assert self.flat_tokens is not None or self.hf_dataset is not None, 'Must provide either flat_tokens or hf_dataset.'
    assert not (self.flat_tokens is not None and self.hf_dataset is not None), 'Should not specify both flat_tokens and hf_dataset.'

  @cached_property
  def training_data(self) -> Union[FlatTokensParams, HuggingFaceDataParams]:
    return self.flat_tokens or self.hf_dataset

def main_contained(config, logger):
  """Main program, which does not access external services except as specified by config.paths or logger."""
  # Use partitionable (and hopefully fusable!) RNG.
  #
  # This is slower in compute time than 'unsafe_rbg' with flag '--xla_tpu_spmd_rng_bit_generator_unsafe=true',
  # but hopefully faster in memory time because it's fusable.
  # TODO: check this is true and if not, provide our own that actually is fusable.
  jax.config.update('jax_threefry_partitionable', True)
  with Mesh(mesh_utils.create_device_mesh([config.mesh.d, config.mesh.t], jax.devices()), ('d', 't')):
    root_rng = jax.random.PRNGKey(config.training.seed)

    loader = get_loader('train', config.training_data, config.training.tokens)
    assert config.model.vocab > loader.max_token_id, f"{config.model.vocab} vs {loader.max_token_id}"
    
    model_dir = os.path.join(config.paths.root_working_dir, config.paths.model_name)
    training_io.mkdir(model_dir)
    state = jax.jit(partial(State.init, config.model))(fold_in_str(root_rng, 'init'))
    state, start_step = training_io.load_checkpoint_if_it_exists(model_dir, state, config.io)

    # Explicitly compile training step, to record XLA HLO graph.
    # See https://bnikolic.co.uk/blog/python/jax/2022/02/22/jax-outputgraph-rev
    c_training_step = training_step.lower(state, jnp.uint32(0), config.model, config.training, loader.load(0)).compile()
    date = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    training_io.save_hlo_svg(os.path.join(model_dir, f'training_step_optimized_hlo_{date}.svg'), c_training_step)

    for step in range(start_step, config.training.steps):
      if step % config.checkpoint_interval == 0 and step > start_step:
        training_io.save_checkpoint(model_dir, step, state, config.io)
      
      # We profile on the second step, because the first step has a long pause for XLA 
      # compilation and initial shuffle buffer loading.
      if jax.process_index() == 0 and step == start_step + 1:
        jax.block_until_ready(state)
        training_io.start_profile()
        profile_start = time.time()

      state, output = c_training_step(state, jnp.uint32(step), loader.load(step))

      # Run profile for two steps, to include data loading time in between them.
      if jax.process_index() == 0 and step == start_step + 2:
        jax.block_until_ready(state)
        profile_duration = time.time() - profile_start
        training_io.stop_profile(model_dir)

        # Print MFU, including (one step of) data loading time.
        print(f"Profile time: {profile_duration}s for 2 steps.")
        model_params = jax.tree.reduce(operator.add, jax.tree.map(lambda w: w.size, state.weights))
        tokens = loader.load(step).targets.size
        print(f'Model params: {model_params:_}')
        print(f'Tokens: {tokens:_}')
        device_flops = training_io.get_flops_per_device()
        num_devices = jax.device_count()
        print(f'MFU (projections only): {100 * (2 * 6 * model_params * tokens / (num_devices * profile_duration)) / device_flops:.2f}% MFU')

      training_io.log(step, logger, output)


@hydra.main(config_path='configs', version_base=None)
def main(config):
  config = jax_extra.make_dataclass_from_dict(Config, config)
  if config.training.queue:
    task = Task.init(project_name='testing', task_name=config.paths.model_name)
    logger = task.get_logger()
    task.execute_remotely(queue_name=config.training.queue)
    task.launch_multi_node(config.num_hosts, wait=True)
    if int(os.environ['RANK']) > 0:
      task.set_system_tags((task.get_system_tags() or []) + ['hidden'])
    jax.distributed.initialize(os.environ['MASTER_ADDR'] + ':' + os.environ['MASTER_PORT'],
                        num_processes=int(os.environ['WORLD_SIZE']),
                        process_id=int(os.environ['RANK']))
  else:
    logger = None
  main_contained(config, logger)


if __name__ == "__main__":
  main()

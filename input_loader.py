"""Input data loading from `flat-tokens` data format.

See `docs/flat-tokens.md` for details on the format.

We support shuffling of the input data, by the following algorithm:
* there are N independent "streams" of data, each of which has disjoint data and is
  shuffled independently.
* within each stream, we fetch a "shuffle buffer" consisting of many "read blocks" of
  data. We shuffle the entire buffer in memory.
* the "read blocks" attached to each shuffle buffer are themselves selected randomly.

This is the standard shuffling used by e.g. Huggingface Datasets. Unlike them, we run
this algorithm _after_ tokenization, so we know exactly at which step number each new
shuffle buffer starts at, allowing us to do instant resumes after job restarts. In our
default recommended configuration, we also recommend a much larger shuffle buffer size
than Huggingface Datasets, which allows for more thorough shuffling, taking advantage
of the fact that a single sequence of tokens uses very little memory compared to e.g.
a single image.

Mosaic's StreamingDatasets library uses a similar algorithm as us, which they call py1b: 
https://docs.mosaicml.com/projects/streaming/en/stable/fundamentals/shuffling.html.
"""

from concurrent.futures import ThreadPoolExecutor
import functools
from typing import Tuple, Union, Optional

from typeguard import typechecked
from shardlib.shardtypes import bool_, pytree_dataclass, u32
import shardlib.shardtypes as shardtypes
import zarr
from dataclasses import dataclass
import jax
import numpy as np
from jax.sharding import PartitionSpec as P
import datetime
import jax

# imports for hf dataloader
import numpy as onp
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from datasets import load_dataset

@dataclass(frozen=True)
class TokenBatchParams:
    """The shape of a token batch."""
    len: int
    batch: int


@pytree_dataclass
class TokenBatch:
    """A batch of tokens, which are typically the input to training."""
    targets: u32['batch/d len']
    is_seq_start: bool_['batch/d len']

        


@dataclass(frozen=True)
class FlatTokensParams:
    filespec: str

    # A "stream" is what's attached to one independent shuffle buffer. There may be multiple
    # independent shuffle buffers, allowing parallelism.
    #
    # A "minipoch" (mini-epoch) is the set of sequences visited by one global refill of shuffle
    # buffers. The last minipoch may be shorter than others, but each stream in the last minipoch
    # must have the same number of read blocks, which must also be an integer.
    #
    # (To minimize discarded data on very small training sets, set streams=1 and make 
    # sequences_per_read_block small.)
    #
    # Shuffling transforms the uint32[num_tokens] into uint32[streams, sequences, len], the
    # "shuffled tokens". We then form batches by a transformation on [streams, sequences].

    streams: int  # Recommended: maximum number of hosts you expect to use.
    read_blocks_per_shuffle_buffer: int  # Recommended: 1 << 10. 4GiB (uncompressed) shuffle buffer.
    sequences_per_read_block: int  # Recommended: (1 << 20) / len. 1MiB (compressed) read block.
    seed: int
    sequence_packing: bool


@dataclass
class _ShuffleBuffer:
    minipoch: int
    buffer: u32['Buflen len']


class ShufflingLoader:
    def __init__(self, split: str, params: FlatTokensParams, token_batch_params: TokenBatchParams):
        self.params = params
        self.token_batch_params = token_batch_params
        self.root = zarr.open_group(params.filespec, mode="r")
        assert split in ["train", "validation"], "Invalid split"
        self.encoded_tokens = self.root[split]["encoded_tokens"]
        self.seq_starts = self.root[split]["seq_starts"]
        self.max_token_id = self.root[split].attrs["max_token_id"]
        assert len(self.encoded_tokens.shape) == 1, "Expected 1D zarr"
        assert self.encoded_tokens.dtype == np.uint32, "Expected uint32 zarr"
        assert len(self.seq_starts.shape) == 1, "Expected 1D zarr"
        assert self.seq_starts.dtype == np.uint64, "Expected uint64 zarr"

        token_count = self.encoded_tokens.shape[0]
        if params.sequence_packing:
            self.seq_count = token_count // token_batch_params.len
        else:
            self.seq_count = self.seq_starts.shape[0] - 1
        
        # Count read blocks. Round it down to a multiple of streams
        read_block_count = self.seq_count // params.sequences_per_read_block
        read_block_count = (read_block_count // params.streams) * params.streams
        self.read_block_count = read_block_count
        assert read_block_count > 0, "Must have at least one read block per stream. Try shrinking streams and sequences_per_read_block."
        self.step_count = (read_block_count * params.sequences_per_read_block) // token_batch_params.batch
        # Count minipochs        
        self.minipoch_count = _div_up(read_block_count, params.streams * params.read_blocks_per_shuffle_buffer)
        self.seq_indices_per_shuffle_buffer = params.read_blocks_per_shuffle_buffer * params.sequences_per_read_block
        # Calculate batch->stream mapping.
        self.batch_indices_per_stream = _div_exact(token_batch_params.batch, params.streams)
        # Calculate which streams and which batch indices this host is responsible for, based on the sharding.
        self.sharding = shardtypes.make_shardings(TokenBatch).targets
        streams = set()
        batch_indices = set()
        for batch_slices, _ in self.sharding.addressable_devices_indices_map((token_batch_params.batch, token_batch_params.len)).values():
            batch_lo, batch_hi, batch_step = batch_slices.indices(token_batch_params.batch)
            for b in range(batch_lo, batch_hi, batch_step):
                batch_indices.add(b)
                streams.add(b // self.batch_indices_per_stream)
        self.shuffle_buffers_by_stream = {stream_index: None for stream_index in streams}
        self.batch_indices = sorted(batch_indices)
        # Shuffle read blocks
        assert read_block_count < 1 << 32, "Too many read blocks. Try growing sequences_per_read_block."
        self.read_block_ordering = _random_permutation(params.seed, read_block_count)


    def load(self, step: int) -> TokenBatch:
        assert step < self.step_count, f"Requested step {step} but dataset only supports {self.step_count} steps at batch size {self.token_batch_params.batch}."
        # Conceptually, we remap IDs as follows:
        # 1. (step, batch_index) -> (stream, seq_index_in_stream)
        # 2. seq_index_in_stream -> (minipoch, seq_index_in_shuffle_buffer)
        #
        # We visit all batch_indices in increasing order. Since the map batch_index->(stream, minipoch)
        # is monotonic (non-decreasing), we can reload the shuffle buffer for a stream whenever
        # we cross to a new minipoch without thrashing back and forth between adjacent minipochs.
        seq_by_batch_index = {}
        for batch_index in self.batch_indices:
            # 1. (step, batch_index) -> (stream, seq_index_in_stream)
            stream = batch_index // self.batch_indices_per_stream
            seq_index_in_stream = step * self.batch_indices_per_stream + (batch_index % self.batch_indices_per_stream)
            # 2. seq_index_in_stream -> (minipoch, seq_index_in_shuffle_buffer)
            minipoch = seq_index_in_stream // self.seq_indices_per_shuffle_buffer
            seq_index_in_shuffle_buffer = seq_index_in_stream % self.seq_indices_per_shuffle_buffer
            shuffle_buffer = self._get_shuffle_buffer(stream, minipoch)
            seq_by_batch_index[batch_index] = shuffle_buffer[seq_index_in_shuffle_buffer]
    
        def get_shard(indexing: Tuple[slice]) -> jax.Array:
            seqlen_slice = indexing[1]
            examples = []
            for batch_index in range(*indexing[0].indices(self.token_batch_params.batch)):
                examples.append(seq_by_batch_index[batch_index][seqlen_slice])
            return np.stack(examples)

        shape = (self.token_batch_params.batch, self.token_batch_params.len)
        encoded_tokens = jax.make_array_from_callback(shape, self.sharding, get_shard)
        return _decode(encoded_tokens)


    def _get_shuffle_buffer(self, stream: int, minipoch: int) -> _ShuffleBuffer:
        if self.shuffle_buffers_by_stream[stream] is None or self.shuffle_buffers_by_stream[stream].minipoch != minipoch:
            self.shuffle_buffers_by_stream[stream] = None  # Free the underlying memory
            blocks_in_shuffle_buffer = self.params.read_blocks_per_shuffle_buffer
            if minipoch == self.minipoch_count - 1:
                blocks_in_shuffle_buffer = (self.read_block_count // self.params.streams) - self.params.read_blocks_per_shuffle_buffer * minipoch
            # We form a mapping:
            #   (stream, minipoch, read_block_in_minipoch) -> sequential_read_block
            # then we map
            #   sequential_read_block -> shuffled_read_block
            # using self.shuffled_read_blocks.
            shuffled_read_block_indices = []
            for read_block_in_minipoch in range(blocks_in_shuffle_buffer):
                sequential_read_block = (minipoch * self.params.read_blocks_per_shuffle_buffer + read_block_in_minipoch) * self.params.streams + stream
                shuffled_read_block = self.read_block_ordering[sequential_read_block]
                shuffled_read_block_indices.append(shuffled_read_block)
            
            # Now load all of the read blocks in parallel.
            def load_read_block(read_block_index: int) -> u32['Buflen len']:
                start_seq = read_block_index * self.params.sequences_per_read_block
                end_seq = start_seq + self.params.sequences_per_read_block
                block_shape = (self.params.sequences_per_read_block, self.token_batch_params.len)
                if self.params.sequence_packing:
                    flat_tokens = self.encoded_tokens[start_seq * self.token_batch_params.len : end_seq * self.token_batch_params.len]
                    return flat_tokens.reshape(block_shape)
                else:
                    seq_starts = self.seq_starts[start_seq : end_seq + 1]
                    flat_tokens = self.encoded_tokens[seq_starts[0] : seq_starts[-1]]
                    # Read the ragged array into a (padded) dense array.
                    #
                    # We pad with 1s, which decode to (0, new_sequence=true).
                    result = np.ones(block_shape, dtype=np.uint32)
                    for i in range(self.params.sequences_per_read_block):
                        start = seq_starts[i]
                        end = seq_starts[i + 1]
                        result[i, :end - start] = flat_tokens[start:end]
                    return result
            
            print(f'[{datetime.datetime.now()}] Loading shuffle buffer')
            # Loading a read block is IO-dominated work, with very little CPU time involved, so we can afford
            # to run a huge number of these in parallel with little concern about thrashing the CPU by having
            # excessively many threads doing CPU-intensive work. At the recommended read block sizing of 1MiB,
            # the memory footprint of a read block is typically bigger than the memory footprint of a CPU thread,
            # so we're also unlikely to waste a significant fraction of memory by having too many threads. In
            # net, allow a lot of threads, potentially way more than we have CPUs! Other overheads will
            # bite us before thread overheads do.
            with ThreadPoolExecutor(max_workers=len(shuffled_read_block_indices)) as executor:
                shuffled_read_blocks = list(executor.map(load_read_block, shuffled_read_block_indices))
            shuffle_buffer = np.concatenate(shuffled_read_blocks, axis=0)
            print(f'[{datetime.datetime.now()}] Finished loading shuffle buffer, {shuffle_buffer.size * 4:_} bytes')
            
            # Actually shuffle it.
            sequences_in_shuffle_buffer = blocks_in_shuffle_buffer * self.params.sequences_per_read_block
            assert shuffle_buffer.shape == (sequences_in_shuffle_buffer, self.token_batch_params.len)
            shuffle_seed = self.params.seed + 1 + minipoch * self.params.streams + stream
            permutation = _random_permutation(shuffle_seed, sequences_in_shuffle_buffer)
            shuffle_buffer = shuffle_buffer[permutation, :]
            self.shuffle_buffers_by_stream[stream] = _ShuffleBuffer(minipoch, shuffle_buffer)
        
        return self.shuffle_buffers_by_stream[stream].buffer

def _div_up(a: int, b: int) -> int:
    return (a + b - 1) // b

def _div_exact(a: int, b: int) -> int:
    assert a % b == 0
    return a // b

@functools.partial(jax.jit, donate_argnums=(0,))
@typechecked
def _decode(encoded_tokens: u32[b'batch/d len']) -> TokenBatch:
    # encoded_tokens encoding:
    #  2*id+1 for the first token in a sequence
    #  2*id for other tokens in the sequence
    return TokenBatch(
        targets = encoded_tokens >> 1,
        is_seq_start = (encoded_tokens & 1) == 1,
    )

def _random_permutation(seed: int, n: int) -> u32['N']:
    """Same as `np.random.Generator.permutation`, but with a guarantee that it will always produce the same results for a given seed."""
    assert n < 1 << 32
    # We do a Fisher-Yates shuffle using the Philox BitGenerator. Unlike the rest of np.random,
    # which is documented as potentially changing between numpy versions or even platforms on
    # the same version, the Philox BitGenerator is documented as stable. Likewise, we also promise
    # not to change the following implementation of the Fisher-Yates shuffle.
    #
    # We calculate the random numbers using `random_uint64() % n` rather than using rejection
    # sampling to generate numbers in range `[0, n)`. (Rejection sampling is more complicated,
    # because we don't know up front how many random numbers we'll need.) Our approach
    # introduces some bias, but it's small: since n<2^32, the bias is at most 2^-32 for each
    # random number generated. We're fine with this.
    randoms = np.random.Philox(seed).random_raw(n) % (np.arange(n, dtype=np.uint64) + 1)
    result = np.arange(n, dtype=np.uint32)
    for i in reversed(range(n)):
        j = randoms[i]
        tmp = result[i]
        result[i] = result[j]
        result[j] = tmp
    return result


@dataclass(frozen=True)
class HuggingFaceDataParams:
    path: str
    tokenizer: str
    num_workers: int
    sequences_packed_per_batch: int
    name: Optional[str] = None

class HuggingFaceDataLoader:
    """
    The HuggingFaceDataLoader is provided for convenience and ease of setup,
    but the flat tokens dataloader is recommended for production use.
    This dataset does not require running the tools/huggingface_to_flat_tokens.py
    to create a flat tokens dataset, and instead streams directly from huggingface.

    This datalaoder will waste tokens if you pack too many sequences into a batch,
    and does not support instant resume to an arbitrary step.
    """
    def __init__(self, split, config: HuggingFaceDataParams, token_batch_params: TokenBatchParams):
        self.tokenizer = AutoTokenizer.from_pretrained(config.tokenizer)
        self.batch_size = token_batch_params.batch
        self.max_seq_len = token_batch_params.len
        self.sharding = shardtypes.make_shardings(TokenBatch).targets
        self.max_token_id = self.tokenizer.vocab_size-1
        assert 0 in self.tokenizer.all_special_ids, "Tokenizer must have a special 0 token"

        # setup an iterator over the dataset
        tokenize = functools.partial(self.tokenizer, padding=False, truncation=False, max_length=None, add_special_tokens=False, return_token_type_ids=False, return_attention_mask=False, return_tensors="np")
        dataset = load_dataset(config.path, config.name, streaming=True, split=split)
        tokenized = dataset.select_columns(["text"]).map(tokenize, input_columns=['text'], remove_columns=["text"])
        dataloader = DataLoader(tokenized, num_workers=config.num_workers, collate_fn=self.collate, drop_last=True, batch_size=config.sequences_packed_per_batch)
        self.iterator = iter(dataloader)

    def collate(self, sequences):
        flat_batch = onp.zeros(self.batch_size * self.max_seq_len, onp.uint32)
        flat_is_start = onp.zeros(self.batch_size * self.max_seq_len, onp.bool_)
        start = 0
        for seq in sequences:
            seq = seq['input_ids'][0]
            end = min(start + len(seq), len(flat_batch))
            flat_is_start[start] = True
            flat_batch[start:end] = seq[:end-start]
            start += len(seq)
            if start >= len(flat_batch):
                break
        shape = (self.batch_size, self.max_seq_len)
        return flat_batch.reshape(shape), flat_is_start.reshape(shape)

    def load(self, step):
        shape = (self.batch_size, self.max_seq_len)
        batch, is_start = next(self.iterator)
        def get_shard(x: jax.Array, indexing: Tuple[slice]) -> jax.Array:
            shard = x[indexing]
            return shard
        tokens = jax.make_array_from_callback(shape, self.sharding, functools.partial(get_shard, batch))
        is_start = jax.make_array_from_callback(shape, self.sharding, functools.partial(get_shard, is_start))
        return TokenBatch(tokens, is_start)

def get_loader(split: str, config: Union[FlatTokensParams, HuggingFaceDataParams], token_batch_params: TokenBatchParams):
    if isinstance(config, FlatTokensParams):
        return ShufflingLoader(split, config, token_batch_params)
    elif isinstance(config, HuggingFaceDataParams):
        return HuggingFaceDataLoader(split, config, token_batch_params)
    else:
        raise ValueError(f"Unknown config type {type(config)}")
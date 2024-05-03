# `flat-tokens` data format

## Introduction

The `flat-tokens` data format is a very simple data format for storing language model training data.
Unlike some other dataset libraries, it supports efficient seeking after job restarts. It also
supports batch size, sequence length, and "sequence packing vs not" being selected at training
time.

It is based on the simplest possible design: a concatenation of all tokens in the dataset, together
with start indices of each sequence.

## Specification

### Flat-tokens array

A *flat-tokens array* is a [`zarr` Group](https://zarr.readthedocs.io/en/stable/) of the following format:

```
arrays: {
  "encoded_tokens": uint32[token_count],
  "seq_starts": uint64[seq_count + 1],
}
attributes: {
  "max_token_id": int32
}
```

That is, it has two arrays, named `encoded_tokens`, `seq_starts`. 

1. The `encoded_tokens` array is a concatenation of all sequences in the dataset into a long array of tokens. 
   There are no padding, beginning-of-sequence, or end-of-sequence tokens included. Tokens are encoded
   as `token_id*2+1` if they are the start of a new sequence, or `token_id*2` if not. The maximum supported `token_id` is `2^31-1`.
2. The `seq_starts` array lists (in increasing order) the indices of the `tokens` array where each 
   sequence starts, plus one final index which equals `token_count`, indicating the end of the final
   sequence.

Additionally, it has one attribute, named `max_token_id`. All decoded `token_id` values in `encoded_tokens`
must be `<= max_token_id`. (This is intended to allow readers to quickly check that their vocabulary size is 
large enough for the dataset.)

### Flat-tokens dataset

A *flat-tokens dataset* is a `zarr` Group with entries "train", "validation", each of which are flat-tokens arrays.

## Example

The token sequences `[[1, 2], [3, 4, 5], [6, 7, 8]]` are represented in a flat-tokens array as:

```
arrays: {
  "tokens": [3, 4, 7, 8, 10, 13, 14, 16],
  "seq_starts": [0, 2, 5, 8],
}
attributes: {
  "max_token_id": 8
}
```

## Discussion

This is the simplest possible format supporting the following features:
* Batch size and sequence length can be chosen at training time. They are not "baked into" the format.
* Data loading can be done with or without sequence packing:
  * Without sequence packing, we consult `seq_starts` to locate the tokens of a particular sequence, e.g. `tokens[seq_starts[1]:seq_starts[2]]` is `[7, 8, 10]`, corresponding to the tokens of sequence 1.
  * With sequence packing, we bypass `seq_starts` and directly consult `tokens`, e.g. for packed sequence length 4, sequence 1 is `tokens[4:8]`, i.e. `[10, 13, 14, 16]`.
* O(1) random access to any sequence, packed or not.
  * This allows you to restart your training job and continue where you left off in the dataset, without retaining any state except for the step or sequence index where you left off.
  * This allows arbitrary shuffling at runtime.
  * Minimal disk seeks ("IO operations" on public clouds) per random access: just one disk seek for sequence-packed random access; just two disk seeks for non-packed random access.

The sequence packing is designed such that no loss masking is required: every single token can be used as a target token. In the above example, if we used packed sequence length 8 (i.e. the whole dataset as one packed sequence),
at training time we'd expand the tokens into the following input and target tokens:

```
{
  "inputs":  [0, 1, 0, 3, 4, 0, 6, 7],
  "targets": [1, 2, 3, 4, 5, 6, 7, 8],
}
```

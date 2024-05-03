# PyTree-zarr checkpoint format

For `seqax` we write checkpoints of JAX PyTrees, in a simple format documented here.

## Specification

The *zarr of a PyTree* is a a [zarr Group](https://zarr.readthedocs.io/en/stable/api/hierarchy.html) with the following elements:
* for each `path, array` in the [flattened PyTree](https://jax.readthedocs.io/en/latest/_autosummary/jax.tree_util.tree_flatten_with_path.html#jax.tree_util.tree_flatten_with_path), the zarr Group contains `array` as a child array, with path equal to `jax.tree_util.keystr(path)`
* additionally there is a zarr [attribute](https://zarr.readthedocs.io/en/stable/api/attrs.html) by name `write_completed` and value `True`.

The zarr of a PyTree may be written to disk with any compression and chunk size settings.

## Discussion

We use `zarr` to support parallel writers from different hosts in a fully-sharded training setup. (Parallel writers in this scenario must choose a chunk size that divides the data size per host, so as to avoid zarr race conditions during writing.) Readers of the checkpoint format do not need to be aware that it was written in parallel, as this is hidden by the zarr abstraction.

We use the `write_completed` attribute to allow parallel writers to support a "two phase commit" protocol: all writers write their data chunks, then wait for a global barrier, then the "leader" writer sets the `write_completed` attribute. This protects readers from reading partially-written checkpoints.

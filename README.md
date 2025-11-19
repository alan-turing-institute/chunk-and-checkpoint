# Chunk and Checkpoint Memory Optimisation

[![CI](https://github.com/alan-turing-institute/chunk-and-checkpoint/actions/workflows/ci.yml/badge.svg)](https://github.com/alan-turing-institute/chunk-and-checkpoint/actions/workflows/ci.yml)


## Installation

```bash
pip install chunkcheck
```

## Usage

`chunkcheck` exports one function: `chunk_and_checkpoint`.
It can be fruitfully used to reduce the peak memory requirement of a programme written using pytorch when the following hold:
- you have one or more input `torch.Tensor`s (`X1`, `X2`, ...) whose first dimension is a "batch" dimension of equal size.
- you wish to compute `f(X1, X2, ...)`, where `f` applies the same operation to each "batch" in (`X1`, `X2`, ...).
- the memory required during intermediate computations in `f` is large compared to the memory required to store (`X1`, `X2`, ...) and the output of `f(X1, X2, ...)`. A canonical example of this kind of function is an MLP with large hidden dimension(s).

Instead of calling `f(X1, X2, ...)`, call `chunk_and_checkpoint(f, X1, X2, ..., chunksize=chunksize)`, for some `int` `chunksize`.
Doing this should substantially reduce peak memory, and increase the computation time by only a small amount for a well-chosen `chunksize`.
`chunk_and_checkpoint` will reduce peak memory further than [`torch.utils.checkpoint.checkpoint`](https://docs.pytorch.org/docs/stable/checkpoint.html) ("activation checkpointing"), the exact amount depends on `chunksize`.

See the docstring for `chunk_and_checkpoint` for more information.
For a more detailed explanation of why this works, and some usage case studies, see our note on arXiv (TODO: write this and link to it).


## Development

Clone the repo and `cd` into the repository.
Then create a virtual environment, enter it, and install all dependencies:

```bash
uv venv
source .venv/bin/activate
uv sync
```

Running the tests:

```bash
pytest -v
```

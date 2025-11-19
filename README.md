# Chunk and Checkpoint Memory Optimisation

[![CI](https://github.com/alan-turing-institute/chunk-and-checkpoint/actions/workflows/ci.yml/badge.svg)](https://github.com/alan-turing-institute/chunk-and-checkpoint/actions/workflows/ci.yml)

Reduce peak memory when training models in PyTorch which require batched operations internally, such as Swin Transformers.

TLDR:

```python
from chunkcheck import chunk_and_checkpoint

...

# There is a really large batch size along dimension 0. `chunk_and_checkpoint`
# substantially reduces peak memory usage. Adjust `chunk_size` to achieve your
# preferred time vs memory tradeoff.
y = chunk_and_checkpoint(f, x1, x2, ..., chunk_size=4, batch_dim=0)

...
```
## Installation

```bash
pip install chunkcheck
```

## Usage

`chunkcheck` exports one function: `chunk_and_checkpoint`.
It can be fruitfully used to reduce the peak memory requirement of a programme written using PyTorch when the following hold:
- You have one or more input `torch.Tensor`s (`x1`, `x2`, ...) whose first dimension is a "batch" dimension of equal size.
- You wish to compute `f(x1, x2, ...)`, where `f` applies the same operation to each "batch" in (`x1`, `x2`, ...).
- The memory required during intermediate computations in `f` is large compared to the memory required to store (`x1`, `x2`, ...) and the output of `f(x1, x2, ...)`. A canonical example of this kind of function is an MLP with large hidden dimension(s).

Instead of calling `f(x1, x2, ...)`, call `chunk_and_checkpoint(f, x1, x2, ..., chunk_size=chunk_size)`, for some `int` `chunk_size`.
Doing this should substantially reduce peak memory, and increase the computation time by only a small amount for a well-chosen `chunk_size`.
`chunk_and_checkpoint` will reduce peak memory further than [`torch.utils.checkpoint.checkpoint`](https://docs.pytorch.org/docs/stable/checkpoint.html) ("activation checkpointing"), the exact amount depends on `chunk_size`.

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

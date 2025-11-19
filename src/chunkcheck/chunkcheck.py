from typing import Callable

import torch
from torch.utils.checkpoint import checkpoint


def chunk_and_checkpoint(
    f: Callable, *xs: torch.Tensor, chunk_size: int, batch_dim: int = 0
) -> torch.Tensor:
    """Compute `f(*xs)` in a memory-efficient manner.

    `chunk_size` controls the time-memory tradeoff. Typically, you want to set
    `chunk_size` just large enough so that this function takes very little more
    time to run than `torch.utils.checkpoint`. To find a good value for
    `chunk_size`, time this function for a few sizes for your use case.

    Args:
        f: A callable.
        xs: A collection of `torch.Tensor`s.
        chunk_size: The number of chunks to divide each element of `xs` into.
        batch_dim: The dimension of each element of `xs` along which to divide.

    """
    # Check that there is at least one positional argument.
    if len(xs) == 0:
        msg = "At least one positional argument required."
        raise ValueError(msg)

    # Verify that xs are all tensors.
    for x in xs:
        if not isinstance(x, torch.Tensor):
            msg = "Arguments must be `torch.Tensor`s."
            raise TypeError(msg)

    # Check that the requested axis is available in all tensors.
    for x in xs:
        if len(x.shape) <= batch_dim:
            msg = "Not all tensors have requested batch axis."
            raise ValueError(msg)

    # Verify that xs have the same length along the batch axis.
    batch_dim_len = xs[0].shape[batch_dim]
    for x in xs[1:]:
        if x.shape[batch_dim] != batch_dim_len:
            msg = "All arguments must have the same batch dim length."
            raise ValueError(msg)

    # Perform checkpointed computation.
    results = []
    n = 0
    while n < batch_dim_len:
        length = min(batch_dim_len - n, chunk_size)
        xs_chunks = [torch.narrow(x, batch_dim, n, length) for x in xs]
        results.append(checkpoint(f, *xs_chunks, use_reentrant=False))
        n = n + chunk_size

    # Concatenate the results and return them.
    return torch.concatenate(results, axis=batch_dim)

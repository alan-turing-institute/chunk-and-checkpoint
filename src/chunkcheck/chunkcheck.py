import torch
from torch.utils.checkpoint import checkpoint


def chunk_and_checkpoint(f, *xs, chunk_size: int, batch_axis=0):
    """Compute f(*xs) in a memory-efficient manner.
    
    `chunk_size` controls the time-memory tradeoff. Typically, you want to set
    `chunk_size` just large enough so that this function takes very little more
    time to run than `torch.utils.checkpoint`. To find a good value for
    `chunk_size`, time this function for a few sizes for your use case.

    Args:
        f: a callable
        xs: a collection of `torch.Tensor`s
        chunk_size: the number of chunks to divide each element of `xs` into.
        batch_axis: the axis of each element of `xs` along which to divide. 
    """

    # Check that there is at least one positional argument.
    if len(xs) == 0:
        raise ValueError("At least one position argument required.")

    # Verify that xs are all tensors.
    for x in xs:
        if type(x) != torch.Tensor:
            raise ValueError("Arguments must be torch Tensors.")

    # Check that the requested axis is available in all tensors.
    for x in xs:
        if len(x.shape) <= batch_axis:
            raise ValueError("Not all tensors have requested batch axis.")

    # Verify that xs have the same length along the batch axis.
    batch_dim_len = xs[0].shape[batch_axis]
    for x in xs[1:]:
        if x.shape[batch_axis] != batch_dim_len:
            raise ValueError("All arguments must have the same batch dim length.")

    # Perform checkpointed computation.
    results = []
    n = 0
    while n < batch_dim_len:
        length = min(batch_dim_len - n, chunk_size)
        xs_chunks = [torch.narrow(x, batch_axis, n, length) for x in xs]
        results.append(checkpoint(f, *xs_chunks, use_reentrant=False))
        n = n + chunk_size

    # Concatenate the results and return them.
    return torch.concatenate(results, axis=batch_axis)

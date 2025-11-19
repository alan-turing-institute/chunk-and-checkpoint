import pytest
import torch
from torch import nn

from chunkcheck import chunk_and_checkpoint


class MLP(torch.nn.Module):
    """Basic MLP class for testing."""

    def __init__(self, dim_in: int, dim_latent: int, dim_out: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim_in, dim_latent, dtype=torch.float64),
            nn.ReLU(),
            nn.Linear(dim_latent, dim_out, dtype=torch.float64),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def test_tensor_value_error() -> None:
    x = torch.randn(5, 4)
    y = torch.randn(4, 4)
    with pytest.raises(ValueError, match=r"At least one positional argument required."):
        chunk_and_checkpoint(lambda x: x, chunk_size=1)
    with pytest.raises(TypeError, match=r"Arguments must be `torch.Tensor`s."):
        chunk_and_checkpoint(lambda x, y: x + y, [], y, chunk_size=1)
    with pytest.raises(
        ValueError, match=r"All arguments must have the same batch dim length."
    ):
        chunk_and_checkpoint(lambda x: x, x, y, chunk_size=1)
    with pytest.raises(ValueError, match=r"Not all tensors have requested batch axis."):
        chunk_and_checkpoint(lambda x: x, x, chunk_size=1, batch_dim=2)


@pytest.mark.parametrize("chunk_size", [1, 2, 3, 4])
@pytest.mark.parametrize("axis", [0, 1])
def test_checkpoint_correctness(chunk_size: int, axis: int) -> None:
    # Set up MLP and argument.
    batch_size = 4
    n = 7
    d = 4
    d_out = 8
    x = torch.randn(batch_size, n, d, requires_grad=True, dtype=torch.float64)
    mlp = MLP(d, 10, d_out)

    # Compute value and gradients in the usual manner + store them.
    y = mlp(x).sum()
    y.backward()
    y_orig = y.detach().clone()
    grads_orig = [p.grad.detach().clone() for p in mlp.parameters()]
    mlp.zero_grad()

    # Compute value and gradients with operation reordering and checkpointing.
    y = chunk_and_checkpoint(mlp, x, chunk_size=chunk_size, batch_dim=axis).sum()
    y.backward()
    y_fuse = y.detach().clone()
    grads_fuse = [p.grad.detach().clone() for p in mlp.parameters()]
    mlp.zero_grad()

    assert torch.isclose(y_orig, y_fuse)
    for g_orig, g_fuse in zip(grads_orig, grads_fuse):
        assert torch.all(torch.isclose(g_orig, g_fuse))

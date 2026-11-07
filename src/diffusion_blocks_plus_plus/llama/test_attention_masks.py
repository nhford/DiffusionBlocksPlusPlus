import torch
import pytest

from .attention_masks import build_diffusion_attention_mask

I = float("-inf")


@pytest.mark.parametrize(
    # fmt: off
    ("clean_mask,clean_positions,noisy_mask,noisy_positions,expected"),
    [(torch.tensor([[1]]),
      torch.tensor([[0]]),
      torch.tensor([[1]]),
      torch.tensor([[1]]),
      torch.tensor([[[[0, I],
                      [0, 0]]]])),
     (torch.tensor([[1, 1]]),
      torch.tensor([[0, 1]]),
      torch.tensor([[0, 1]]),
      torch.tensor([[1, 2]]),
      torch.tensor([[[[0, I, I, I],
                      [0, 0, I, I],
                      [I, I, 0, I],
                      [0, 0, I, 0]]]]))],
    # fmt: on
)
def test_build_diffusion_attention_mask(
    clean_mask: torch.Tensor,
    clean_positions: torch.Tensor,
    noisy_mask: torch.Tensor,
    noisy_positions: torch.Tensor,
    expected: torch.Tensor,
):
    actual = build_diffusion_attention_mask(
        clean_mask,
        clean_positions,
        noisy_mask,
        noisy_positions,
    )
    torch.testing.assert_close(actual, expected)

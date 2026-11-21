import math

import pytest
import torch

from diffusion_blocks_plus_plus.llama.samplers import sample


@pytest.mark.parametrize("sampler", ["euler", "heun", "midpoint"])
def test_sampler_convergence(sampler):
    # dx/dt = x, integrated backwards from t=0 to t=-1, has x(-1)=exp(-1).
    def error(num_steps):
        result = sample(
            torch.ones(1, dtype=torch.float64),
            torch.linspace(0, -1, num_steps + 1).tolist(),
            lambda x, t: x,
            sampler,
        )
        return abs(result.item() - math.exp(-1))

    ratio = error(20) / error(40)
    expected_ratio = 2 if sampler == "euler" else 4
    assert ratio == pytest.approx(expected_ratio, rel=0.1)


@pytest.mark.parametrize(
    ("sampler", "expected_calls", "expected_result"),
    [
        ("euler", [(2.0, 1.0)], -4.0),
        ("heun", [(2.0, 1.0), (-4.0, -1.0)], 4.0),
        ("midpoint", [(2.0, 1.0), (-1.0, 0.0)], 4.0),
    ],
)
def test_sampler_evaluation_points(sampler, expected_calls, expected_result):
    calls = []

    def ode(x, t):
        calls.append((x.item(), t))
        return x + t

    initial = torch.tensor([2.0])
    result = sample(initial, [1.0, -1.0], ode, sampler)
    assert calls == expected_calls
    torch.testing.assert_close(result, torch.tensor([expected_result]))
    assert result is initial


def test_sampler_rejects_unknown_method():
    with pytest.raises(ValueError, match="Unknown sampler"):
        sample(torch.zeros(1), [1.0, 0.0], lambda x, t: x, "unknown")


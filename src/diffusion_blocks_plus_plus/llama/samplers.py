import itertools
import math
from collections.abc import Callable, Sequence

import torch
from tqdm import tqdm


def sample(
    initial_embeddings: torch.Tensor,
    lambdas: Sequence[float],
    ode: Callable[[torch.Tensor, float], torch.Tensor],
    sampler: str = "karras_euler",
) -> torch.Tensor:
    """Integrate embeddings over a log-SNR (lambda) grid.

    Here lambda = log(SNR) = -log(sigma), so lambdas should be increasing
    as noise decreases. The ODE callback returns dx/dsigma at the given
    lambda, with sigma = exp(-lambda). Model selection and
    denoising predictions belong to the caller. Updates modify the supplied
    embeddings in place, matching the original evaluation loop.
    """
    if sampler in {"dpmpp_2m", "dpmpp_2s"}:
        return sample_dpmpp(initial_embeddings, lambdas, ode, sampler)

    if sampler in {"karras_euler", "karras_heun", "karras_midpoint"}:
        return sample_karras(initial_embeddings, lambdas, ode, sampler)

    raise ValueError(f"Unknown sampler: {sampler}")


def sample_karras(
    initial_embeddings: torch.Tensor,
    lambdas: Sequence[float],
    ode: Callable[[torch.Tensor, float], torch.Tensor],
    sampler: str,
) -> torch.Tensor:
    """Integrators for the EDM (Karras) formulation, stepping in sigma."""
    embeddings = initial_embeddings
    for lam, next_lam in tqdm(
        list(itertools.pairwise(lambdas)),
        desc="Denoise Step",
        leave=False,
    ):
        # Note that we do this because the original paper uses the EDM Karras
        # formulation.
        d_stdev = math.exp(-next_lam) - math.exp(-lam)
        slope = ode(embeddings, lam)
        if sampler == "karras_euler":
            embeddings += slope * d_stdev
        elif sampler == "karras_heun":
            predictor = embeddings + slope * d_stdev
            endpoint_slope = ode(predictor, next_lam)
            embeddings += (slope + endpoint_slope) * (d_stdev / 2)
        elif sampler == "karras_midpoint":
            midpoint = embeddings + slope * (d_stdev / 2)
            midpoint_slope = ode(midpoint, -math.log(math.exp(-lam) + d_stdev / 2))
            embeddings += midpoint_slope * d_stdev
        else:
            raise ValueError(f"Unknown sampler: {sampler}")
    return embeddings


def sample_dpmpp(
    initial_embeddings: torch.Tensor,
    lambdas: Sequence[float],
    ode: Callable[[torch.Tensor, float], torch.Tensor],
    sampler: str,
) -> torch.Tensor:
    """See https://arxiv.org/pdf/2211.01095"""

    def data_prediction(x: torch.Tensor, lam: float) -> torch.Tensor:
        return x - math.exp(-lam) * ode(x, lam)

    x = initial_embeddings
    prev_data_pred: torch.Tensor | None = None
    prev_h: float | None = None

    for lam_i, lam_next in tqdm(
        list(itertools.pairwise(lambdas)),
        desc="Denoise Step",
        leave=False,
    ):
        h = lam_next - lam_i
        sigma_ratio = math.exp(-h)
        data_pred = data_prediction(x, lam_i)

        if sampler == "dpmpp_2s":
            r = 0.5
            lam_s = lam_i + r * h
            u = math.exp(-r * h) * x - (math.exp(-r * h) - 1) * data_pred
            data_pred_s = data_prediction(u, lam_s)
            D = (1 - 1 / (2 * r)) * data_pred + (1 / (2 * r)) * data_pred_s
        elif sampler == "dpmpp_2m":
            if prev_data_pred is None or prev_h is None:
                D = data_pred
            else:
                r = prev_h / h
                D = (1 + 1 / (2 * r)) * data_pred - (1 / (2 * r)) * prev_data_pred
            prev_data_pred, prev_h = data_pred, h
        else:
            raise ValueError(f"Unknown sampler: {sampler}")

        x = sigma_ratio * x - (math.exp(-h) - 1) * D
    return x

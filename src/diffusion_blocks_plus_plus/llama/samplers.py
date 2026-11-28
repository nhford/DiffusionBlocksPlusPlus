import itertools
import math
from collections.abc import Callable, Sequence

import torch
from tqdm import tqdm


def sample(
    initial_embeddings: torch.Tensor,
    timesteps: Sequence[float],
    ode: Callable[[torch.Tensor, float], torch.Tensor],
    sampler: str = "euler",
) -> torch.Tensor:
    """Integrate embeddings over a log-noise timestep grid.

    The ODE callback returns dx/dt for t = log(sigma). Model selection and
    denoising predictions belong to the caller. Updates modify the supplied
    embeddings in place, matching the original evaluation loop.
    """
    if sampler in {"dpmpp_2m", "dpmpp_2s"}:
        return sample_dpmpp(initial_embeddings, timesteps, ode, sampler)

    embeddings = initial_embeddings
    for timestep, next_timestep in tqdm(
        list(itertools.pairwise(timesteps)),
        desc="Denoise Step",
        leave=False,
    ):
        # Note that we do this because the original paper uses the EDM Karras
        # formulation.
        d_stdev = math.exp(next_timestep) - math.exp(timestep)
        slope = ode(embeddings, timestep)
        if sampler == "euler":
            embeddings += slope * d_stdev
        elif sampler == "heun":
            predictor = embeddings + slope * d_stdev
            endpoint_slope = ode(predictor, next_timestep)
            embeddings += (slope + endpoint_slope) * (d_stdev / 2)
        elif sampler == "midpoint":
            midpoint = embeddings + slope * (d_stdev / 2)
            midpoint_slope = ode(midpoint, timestep + d_stdev / 2)
            embeddings += midpoint_slope * d_stdev
        else:
            raise ValueError(f"Unknown sampler: {sampler}")
    return embeddings


def sample_dpmpp(
    initial_embeddings: torch.Tensor,
    timesteps: Sequence[float],
    ode: Callable[[torch.Tensor, float], torch.Tensor],
    sampler: str,
) -> torch.Tensor:
    """See https://arxiv.org/pdf/2211.01095"""

    def data_prediction(x: torch.Tensor, t: float) -> torch.Tensor:
        return x - math.exp(t) * ode(x, t)

    x = initial_embeddings
    prev_data_pred: torch.Tensor | None = None
    prev_h: float | None = None

    for t_i, t_next in tqdm(
        list(itertools.pairwise(timesteps)),
        desc="Denoise Step",
        leave=False,
    ):
        lambda_i, lambda_next = -t_i, -t_next
        h = lambda_next - lambda_i
        sigma_ratio = math.exp(t_next - t_i)
        data_pred = data_prediction(x, t_i)

        if sampler == "dpmpp_2s":
            r = 0.5
            t_s = -(lambda_i + r * h)
            u = math.exp(t_s - t_i) * x - (math.exp(-r * h) - 1) * data_pred
            data_pred_s = data_prediction(u, t_s)
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

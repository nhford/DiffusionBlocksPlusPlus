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
    if sampler not in {"euler", "heun", "midpoint"}:
        raise ValueError(f"Unknown sampler: {sampler}")

    embeddings = initial_embeddings
    for timestep, next_timestep in tqdm(
        list(itertools.pairwise(timesteps)),
        desc="Denoise Step",
        leave=False,
    ):
        d_stdev = math.exp(next_timestep) - math.exp(timestep)
        slope = ode(embeddings, timestep)
        if sampler == "euler":
            embeddings += slope * d_stdev
        elif sampler == "heun":
            predictor = embeddings + slope * d_stdev
            endpoint_slope = ode(predictor, next_timestep)
            embeddings += (slope + endpoint_slope) * (d_stdev / 2)
        else:  # midpoint
            midpoint = embeddings + slope * (d_stdev / 2)
            midpoint_slope = ode(midpoint, timestep + d_stdev / 2)
            embeddings += midpoint_slope * d_stdev
    return embeddings

import itertools
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

    Euler uses one ODE evaluation per interval. Heun and midpoint use two,
    evaluating the second slope at the endpoint predictor or halfway predictor
    respectively. All predictor times are in log-noise coordinates.
    """
    if sampler not in {"euler", "heun", "midpoint"}:
        raise ValueError(f"Unknown sampler: {sampler}")

    embeddings = initial_embeddings
    for timestep, next_timestep in tqdm(
        list(itertools.pairwise(timesteps)),
        desc="Denoise Step",
        leave=False,
    ):
        dt = next_timestep - timestep
        slope = ode(embeddings, timestep)
        if sampler == "euler":
            embeddings += slope * dt
        elif sampler == "heun":
            predictor = embeddings + slope * dt
            endpoint_slope = ode(predictor, next_timestep)
            embeddings += (slope + endpoint_slope) * (dt / 2)
        else:  # midpoint
            midpoint = embeddings + slope * (dt / 2)
            midpoint_slope = ode(midpoint, timestep + dt / 2)
            embeddings += midpoint_slope * dt
    return embeddings

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
    """
    if sampler != "euler":
        raise ValueError(f"Unknown sampler: {sampler}")

    embeddings = initial_embeddings
    for timestep, next_timestep in tqdm(
        list(itertools.pairwise(timesteps)),
        desc="Denoise Step",
        leave=False,
    ):
        dt = next_timestep - timestep
        embeddings += ode(embeddings, timestep) * dt
    return embeddings

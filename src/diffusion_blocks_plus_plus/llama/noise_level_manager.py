import math

import numpy as np
import torch
from scipy.stats import norm


class NoiseLevelDistribution:
    """Defines a target noise distribution for a DiffusionBlocks model.

    This class specifically describes a log-normal noise distribution.

    Args:
        p_mean: The mean log noise level.
        p_std: The standard deviation of the log noise level.
        gamma: The margin by which log noise levels are expanded in train.
        stdev_min: The hard lower cutoff for noise levels.
        stdev_max: The hard upper cutoff for noise levels.
    """

    def __init__(
        self,
        p_mean=-1.2,
        p_std=1.2,
        gamma=0.05,
        stdev_min=0.002,
        stdev_max=80.0,
    ):
        self.p_mean = p_mean
        self.p_std = p_std
        self.gamma = gamma
        self.stdev_min = stdev_min
        self.stdev_max = stdev_max

    def get_training_stdevs(
        self, num_batches: int, block_stdev_min: float, block_stdev_max: float, device
    ):
        """Get a batch of randomly sampled noise standard deviations for a block
        during training.

        Args:
            num_batches: The number of samples to take.
            stdev_min: The minimum noise level assigned to the block at test
                time.
            stdev_max: The maximum noise level assigned to the block at test
                time.
            device: The device to place the output in.

        Returns:
            A tensor of shape `(num_batches,)` with the sampled noise levels.

        Adapted from the DiffusionBlocks repository.
        """
        log_stdev_min = math.log(block_stdev_min)
        log_stdev_max = math.log(block_stdev_max)

        log_stdev_range = log_stdev_max - log_stdev_min

        # Expand the range as per the DiffusionBlocks paper.
        log_stdev_min -= log_stdev_range * self.gamma
        log_stdev_max += log_stdev_range * self.gamma

        z_min = (log_stdev_min - self.p_mean) / self.p_std
        z_max = (log_stdev_max - self.p_mean) / self.p_std

        percentile_min = norm.cdf(z_min).item()
        percentile_max = norm.cdf(z_max).item()

        percentile = (
            torch.rand((num_batches,), device=device)
            * (percentile_max - percentile_min)
            + percentile_min
        )

        z = math.sqrt(2) * torch.erfinv(percentile * 2 - 1)
        log_stdev = self.p_mean + self.p_std * z
        stdev = log_stdev.exp()
        return stdev

    def get_block_stdevs(
        self,
        num_layers: int,
    ) -> list[float]:
        """From src/diffusion_blocks_plus_plus/vit/dblock_modules.py"""
        cdf_min = norm.cdf((np.log(self.stdev_min) - self.p_mean) / self.p_std)
        cdf_max = norm.cdf((np.log(self.stdev_max) - self.p_mean) / self.p_std)
        block_stdevs = []
        for i in range(num_layers + 1):
            p = cdf_min + (cdf_max - cdf_min) * (i / num_layers)
            stdev = np.exp(self.p_mean + self.p_std * norm.ppf(p))
            block_stdevs.append(stdev)
        return block_stdevs

    def get_discrete_stdevs(
        self,
        num_steps: int,
    ) -> list[float]:
        """From src/diffusion_blocks_plus_plus/vit/dblock_modules.py"""
        log_stdev_min = np.log(self.stdev_min)
        log_stdev_max = np.log(self.stdev_max)
        cdf_min = norm.cdf((log_stdev_min - self.p_mean) / self.p_std)
        cdf_max = norm.cdf((log_stdev_max - self.p_mean) / self.p_std)
        cdf_points = np.linspace(cdf_min, cdf_max, num_steps)
        stdevs = np.exp(self.p_mean + self.p_std * norm.ppf(cdf_points))
        stdevs = torch.tensor(stdevs, dtype=torch.float32)
        stdevs = torch.flip(stdevs, dims=[0])
        stdevs = stdevs.tolist()
        return stdevs

import gc
import warnings

import psutil
import torch
from torch.optim import Optimizer

from diffusion_blocks_plus_plus.llama.model import LlamaDiTForCausalLM


def format_mem(bytes: int):
    if bytes >= 1e12:
        return f"{bytes / 1e12:#.4G} TB"
    elif bytes >= 1e9:
        return f"{bytes / 1e9:#.4G} GB"
    elif bytes >= 1e6:
        return f"{bytes / 1e6:#.4G} MB"
    elif bytes >= 1e3:
        return f"{bytes / 1e3:#.4G} KB"
    else:
        return f"{bytes} B"


class MemoryTracker:
    """Tracker for the major components of memory usage during training.

    This class is specifically designed for LlamaDitForCausalLM.
    To use this class, call the measure_mem_* methods at the appropriate locations.

    Args:
        model: The model to track memory usage for.
        optimizer: The optimizer to track memory usage for.
        device: The device where the memory is stored.
    """

    def __init__(
        self, model: LlamaDiTForCausalLM, optimizer: Optimizer, device: torch.device
    ):
        self.model = model
        self.optimizer = optimizer
        self.usage_dict: dict[str, int | None] = {
            "weights": None,
            "activations": None,
            "gradients": None,
            "optimizer": None,
        }
        self.forward_initial_mem: int | None = None
        self.device = device

    def reset(self):
        for key in self.usage_dict.keys():
            self.usage_dict[key] = None
        self.forward_initial_mem = None

    def _current_memory_used(self):
        if self.device.type == "cuda":
            return torch.cuda.memory_allocated(self.device)
        elif self.device.type == "cpu":
            warnings.warn(
                "Measuring system RAM because device=cpu. Results may be inaccurate."
            )
            gc.collect()
            return psutil.Process().memory_info().rss
        else:
            warnings.warn("device not supported.")
            return 0

    def _assert_unused(self, value: int | None):
        if value is not None:
            warnings.warn("Forgot to call reset() between measurements.")

    def measure_mem_weights(self):
        self._assert_unused(self.usage_dict["weights"])
        if all(block is None for block in self.model.blocks_loaded):
            warnings.warn(
                "measure_mem_weights was called with no blocks loaded. This is probably incorrect."
            )
        self.usage_dict["weights"] = sum(
            param.numel() * param.element_size()
            for param in self.model.loaded_parameters()
        )

    def begin_forward(self):
        """Begin tracking the forward pass.

        Call this function before loading the batch into the device."""

        self._assert_unused(self.forward_initial_mem)
        self.forward_initial_mem = self._current_memory_used()

    def measure_mem_act(self):
        self._assert_unused(self.usage_dict["activations"])
        if self.forward_initial_mem is None:
            warnings.warn("Forgot to call begin_forward first")
            return
        self.usage_dict["activations"] = (
            self._current_memory_used() - self.forward_initial_mem
        )

    def measure_mem_grad(self):
        self._assert_unused(self.usage_dict["gradients"])
        self.usage_dict["gradients"] = sum(
            param.grad.numel() * param.grad.element_size()
            for param in self.model.loaded_parameters()
            if param.grad is not None
        )
        if self.usage_dict["gradients"] == 0:
            warnings.warn(
                "Got mem_grad = 0. Did you forget to measure between loss.backward() and optimizer.step()?"
            )

    def measure_mem_optim(self):
        self._assert_unused(self.usage_dict["optimizer"])
        self.usage_dict["optimizer"] = 0
        for optimizer_params in self.optimizer.state.values():
            if not isinstance(optimizer_params, dict):
                continue
            for state_entry in optimizer_params.values():
                self.usage_dict["optimizer"] += (
                    state_entry.numel() * state_entry.element_size()
                )
        if self.usage_dict["optimizer"] == 0:
            warnings.warn(
                "Got mem_optim = 0. Did you forget to measure after optimizer.step()?"
            )

    def as_dict(self):
        if any(usage is None for usage in self.usage_dict.values()):
            warnings.warn(
                f"Incomplete measurements for memory usage: {self.usage_dict}"
            )
        prefix = "mem_"
        total_memory = sum(
            usage for usage in self.usage_dict.values() if usage is not None
        )
        return {
            **{(prefix + key): value for key, value in self.usage_dict.items()},
            f"{prefix}total": total_memory,
        }

    def tqdm_dict(self):
        return {key: format_mem(value) for key, value in self.as_dict().items()}

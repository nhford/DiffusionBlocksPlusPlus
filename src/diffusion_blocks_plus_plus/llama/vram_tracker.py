import gc
import warnings

import psutil
import torch
from torch.optim import Optimizer

from diffusion_blocks_plus_plus.llama.model import LlamaDiTForCausalLM


def format_mem(bytes: int):
    if bytes >= 1e12:
        return f"{bytes/1e12:.4} TB"
    elif bytes >= 1e9:
        return f"{bytes/1e9:.4} GB"
    elif bytes >= 1e6:
        return f"{bytes/1e6:.4} MB"
    elif bytes >= 1e3:
        return f"{bytes/1e3:.3} KB"
    else:
        return f"{bytes} B"


class MemoryTracker:
    def __init__(
        self, model: LlamaDiTForCausalLM, optimizer: Optimizer, device: torch.device
    ):
        self.model = model
        self.optimizer = optimizer
        self.vram_model: int | None = None
        self.vram_act: int | None = None
        self.vram_grad: int | None = None
        self.vram_optim: int | None = None
        self.forward_initial_vram: int | None = None
        self.device = device

    def reset(self):
        self.vram_model = None
        self.vram_act = None
        self.vram_grad = None
        self.vram_optim = None
        self.forward_initial_vram = None

    def _current_memory_used(self):
        if self.device.type == "cuda":
            return torch.cuda.memory_allocated(self.device)
        elif self.device.type == "cpu":
            warnings.warn("Measuring system RAM because device=cpu. Results may be inaccurate.")
            gc.collect()
            return psutil.Process().memory_info().rss
        else:
            warnings.warn("device not supported.")
            return 0

    def _assert_unused(self, value: int | None):
        if value is not None:
            warnings.warn("Forgot to call reset() between measurements.")

    def measure_vram_model(self):
        self._assert_unused(self.vram_model)
        if all(block is None for block in self.model.blocks_loaded):
            warnings.warn(
                "measure_vram_model was called with no blocks loaded. This is probably incorrect."
            )
        self.vram_model = sum(
            param.numel() * param.element_size()
            for param in self.model.loaded_parameters()
        )

    def begin_forward(self):
        self._assert_unused(self.forward_initial_vram)
        self.forward_initial_vram = self._current_memory_used()

    def measure_vram_act(self):
        self._assert_unused(self.vram_act)
        if self.forward_initial_vram is None:
            warnings.warn(
                "measure_vram_model was called with no blocks loaded. This is probably incorrect."
            )
            return
        self.vram_act = self._current_memory_used() - self.forward_initial_vram

    def measure_vram_grad(self):
        self._assert_unused(self.vram_grad)
        self.vram_grad = sum(
            param.grad.numel() * param.grad.element_size()
            for param in self.model.loaded_parameters()
            if param.grad is not None
        )
        if self.vram_grad == 0:
            warnings.warn(
                "Got vram_grad = 0. Did you forget to measure between loss.backward() and optimizer.step()?"
            )

    def measure_vram_optim(self):
        self._assert_unused(self.vram_optim)
        self.vram_optim = 0
        for optimizer_params in self.optimizer.state.values():
            if not isinstance(optimizer_params, dict):
                continue
            for state_entry in optimizer_params.values():
                self.vram_optim += state_entry.numel() * state_entry.element_size()
        if self.vram_grad == 0:
            warnings.warn(
                "Got vram_optim = 0. Did you forget to measure after optimizer.step()?"
            )

    def as_dict(self):
        if (
            self.vram_model is None
            or self.vram_act is None
            or self.vram_grad is None
            or self.vram_optim is None
        ):
            warnings.warn("Incomplete measurements for memory usage.")
            # TODO: Make this graceful
            return dict()
        prefix = "vram_"
        return {
            f"{prefix}model": self.vram_model,
            f"{prefix}activations": self.vram_act,
            f"{prefix}gradients": self.vram_grad,
            f"{prefix}optimizer": self.vram_optim,
            f"{prefix}total": self.vram_model
            + self.vram_act
            + self.vram_grad
            + self.vram_optim,
        }

    def tqdm_dict(self):
        return {key: format_mem(value) for key, value in self.as_dict().items()}

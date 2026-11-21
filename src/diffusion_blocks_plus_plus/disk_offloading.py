"""
Utility script for backing model parameters with memory-mapped storage
"""

from os import PathLike

import torch
from transformers.modeling_utils import PreTrainedModel


@torch.no_grad
def map_to_file(model: PreTrainedModel, path: str | PathLike) -> int:
    # if True:
    #     model.to("cpu")
    #     return 0
    state_dict = model.state_dict(keep_vars=True)

    if len(state_dict) != len({id(t) for t in state_dict.values()}):
        raise ValueError("Duplicates in state dict")

    total_size = sum(
        (weight.numel() * weight.element_size() + 63) // 64 * 64
        for weight in state_dict.values()
    )
    with open(path, "wb") as f:
        f.truncate(total_size)

    storage = torch.UntypedStorage.from_file(str(path), shared=True, nbytes=total_size)
    assert isinstance(storage, torch.UntypedStorage)

    # Replace the backing storage
    offset = 0
    for name, state in state_dict.items():
        new_state = torch.empty_like(state, device="cpu").contiguous()

        new_state.set_(
            storage,
            offset // new_state.element_size(),
            new_state.size(),
            new_state.stride(),
        )

        if not state.is_meta:
            new_state.copy_(state)

        state_dict[name] = new_state

        offset += state.numel() * state.element_size()
        offset = (offset + 63) // 64 * 64

    assert offset == total_size

    model.load_state_dict(state_dict, assign=True)

    return total_size

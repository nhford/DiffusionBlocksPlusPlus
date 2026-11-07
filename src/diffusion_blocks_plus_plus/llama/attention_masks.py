import torch


def build_diffusion_attention_mask(
    clean_mask: torch.Tensor,
    clean_positions: torch.Tensor,
    noisy_mask: torch.Tensor,
    noisy_positions: torch.Tensor,
    dtype=torch.float,
) -> torch.Tensor:
    """Build the attention mask for auto-regressive DiffusionBlocks input
    given metadata about a clean and noisy sequence.

    Throughout the rest of this documentation, let
    `B` be the number of sequences in the current batch,
    `S_c` be the length of the clean sequence being processed, and
    `S_n` be the length of the noisy sequence being processed.

    This function builds a rank-4 attention mask that allows tokens to
    attend to themselves and unmasked tokens with lower indices, where 0
    represents attention values that should be allowed, and negative
    infinity represents attention values that are masked out.

    For more information on rank-4 attention masks, see
    https://github.com/huggingface/transformers/pull/27539

    Args:
    clean_mask: A shape `(B, S_c)` tensor containing 1 at locations in the
        clean sequence with non-padding tokens, and 0 at locations with
        padding tokens.
    clean_positions: A shape `(B, S_c)` tensor containing position ids for
        the clean sequence.
    noisy_mask: A shape `(B, S_n)` tensor containing 1 at locations in the
        noisy sequence with non-padding tokens, and 0 at locations with
        padding.
    noisy_positions: A shape `(B, S_n)` tensor containing position ids for
        the noisy sequence.
    dtype: The dtype of the attention matrix.


    Returns:
        A tensor of shape `(B, 1, (S_c + S_n), (S_c + S_n))` with the
        attention mask.
    """
    if not (
        len(clean_mask.shape) == 2
        and len(clean_positions.shape) == 2
        and len(noisy_mask.shape) == 2
        and len(noisy_positions.shape) == 2
        and clean_mask.shape[0]
        == clean_positions.shape[0]
        == noisy_mask.shape[0]
        == noisy_positions.shape[0]
        and clean_mask.shape[1] == clean_positions.shape[1]
        and noisy_mask.shape[1] == noisy_positions.shape[1]
    ):
        raise ValueError(
            "Invalid shapes:\n"
            f"{clean_mask.shape=}\n"
            f"{clean_positions.shape=}\n"
            f"{noisy_mask.shape=}\n"
            f"{noisy_positions.shape=}"
        )

    if not (
        clean_mask.device
        == clean_positions.device
        == noisy_mask.device
        == noisy_positions.device
    ):
        raise ValueError(
            "All tensors must be on the same device. Got\n"
            f"{clean_mask.device=}\n"
            f"{clean_positions.device=}\n"
            f"{noisy_mask.device=}\n"
            f"{noisy_positions.device=}"
        )

    clean_mask = clean_mask.bool()
    noisy_mask = noisy_mask.bool()

    device = clean_mask.device
    B = clean_mask.shape[0]
    S_c = clean_mask.shape[1]
    S_n = noisy_mask.shape[1]

    clean_query_idx = clean_positions[:, None, :, None]
    clean_key_idx = clean_positions[:, None, None, :]

    noisy_query_idx = noisy_positions[:, None, :, None]
    noisy_key_idx = noisy_positions[:, None, None, :]

    # Compute the four quadrants of the attention mask.
    clean_to_clean = clean_query_idx > clean_key_idx
    clean_to_clean = clean_to_clean & clean_mask[:, None, None, :]
    clean_to_clean = clean_to_clean & clean_mask[:, None, :, None]
    clean_to_clean = (
        clean_to_clean
        | torch.eye(S_c, dtype=torch.bool, device=device)[None, None, :, :]
    )
    clean_to_clean = clean_to_clean.broadcast_to((B, 1, S_c, S_c))

    noisy_to_clean = noisy_query_idx > clean_key_idx
    noisy_to_clean = noisy_to_clean & clean_mask[:, None, None, :]
    noisy_to_clean = noisy_to_clean & noisy_mask[:, None, :, None]
    noisy_to_clean = noisy_to_clean.broadcast_to((B, 1, S_n, S_c))

    noisy_to_noisy = torch.eye(S_n, dtype=torch.bool, device=device)[None, None, :, :]
    noisy_to_noisy = noisy_to_noisy.broadcast_to((B, 1, S_n, S_n))

    clean_to_noisy = torch.zeros((B, 1, S_c, S_n), dtype=torch.bool, device=device)

    # Convert the quadrants into an attention matrix.
    attention_mask = torch.cat(
        [
            torch.cat([clean_to_clean, clean_to_noisy], dim=3),
            torch.cat([noisy_to_clean, noisy_to_noisy], dim=3),
        ],
        dim=2,
    )
    attention_mask = torch.where(attention_mask, 0, float("-inf")).to(dtype)

    return attention_mask


def build_train_attention_mask(token_mask: torch.Tensor):
    """Build the attention mask for auto-regressive DiffusionBlocks
    training.

    Throughout the rest of this documentation, let
    `B` be the number of sequences in the current batch, and
    `S` be the length of the token sequence being processed.

    See `build_diffusion_attention_mask` for nore information.

    Args:
        token_mask: A shape `(B, S)` tensor containing 1 at locations
            with non-padding tokens, and 0 at locations with padding tokens.

    Returns:
        A tensor of shape `(B, 1, (S * 2 - 2), (S * 2 - 2))` with the
        attention mask.
    """
    if len(token_mask.shape) != 2:
        raise ValueError(f"Invalid token mask shape: {token_mask.shape=}")

    device = token_mask.device
    num_batches, seq_len = token_mask.shape

    clean_mask = token_mask[:, :-1]
    noisy_mask = token_mask[:, 1:]
    clean_positions = torch.arange(0, seq_len - 1, device=device).broadcast_to(
        num_batches, seq_len - 1
    )
    noisy_positions = torch.arange(1, seq_len, device=device).broadcast_to(
        num_batches, seq_len - 1
    )

    return build_diffusion_attention_mask(
        clean_mask=clean_mask,
        clean_positions=clean_positions,
        noisy_mask=noisy_mask,
        noisy_positions=noisy_positions,
    )

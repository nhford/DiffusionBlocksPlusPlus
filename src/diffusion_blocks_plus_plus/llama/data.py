import datasets
import torch
from datasets import dataset_dict, load_dataset
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, LlamaTokenizerFast

from diffusion_blocks_plus_plus.llama.config import LlamaDataConfig

from .attention_masks import (
    build_bd3lm_train_attention_mask,
    build_train_attention_mask,
)
from .noise_level_manager import NoiseLevelDistribution


def trivial_random_dataset(num_samples, sequence_len):
    return datasets.Dataset.from_dict(
        {
            "input_ids": torch.randint(0, 2, (num_samples, sequence_len)),
            "attention_mask": torch.ones(num_samples, sequence_len, dtype=torch.long),
        }
    )


def predictable_dataset(num_samples, sequence_len):
    block_size = 4
    block_count = sequence_len + (block_size - 1) // block_size
    input_ids = torch.randint(0, 2, (num_samples, block_count, 1))
    input_ids = input_ids.broadcast_to((num_samples, block_count, block_size))
    input_ids = input_ids.reshape((num_samples, -1))
    input_ids = input_ids[:, :sequence_len]

    return datasets.Dataset.from_dict(
        {
            "input_ids": input_ids,
            "attention_mask": torch.ones(num_samples, sequence_len, dtype=torch.long),
        }
    )


def build_timesteps(
    inputs_embeds: torch.Tensor, timestep: torch.Tensor | float, mode: str
) -> torch.Tensor:
    """Condition concatenated [clean, noisy] tokens on log-noise timesteps."""
    batch_size, sequence_length = inputs_embeds.shape[:2]
    timesteps = (
        torch.as_tensor(
            timestep, device=inputs_embeds.device, dtype=inputs_embeds.dtype
        )
        .reshape(-1, 1)
        .expand(batch_size, 1)
    )
    if mode == "uniform":
        return timesteps
    if mode == "per-token":
        clean = inputs_embeds.new_zeros((batch_size, sequence_length // 2))
        return torch.cat([clean, timesteps.expand_as(clean)], dim=1)
    raise ValueError(f"Unknown timestep conditioning mode: {mode}")


def load_data(data_config: LlamaDataConfig):
    tokenizer = AutoTokenizer.from_pretrained(data_config.tokenizer_config.name)
    assert isinstance(tokenizer, LlamaTokenizerFast)
    tokenizer.padding_side = data_config.tokenizer_config.padding_side
    tokenizer.add_bos_token = data_config.tokenizer_config.add_bos_token
    tokenizer.add_eos_token = data_config.tokenizer_config.add_eos_token

    def tokenize_function(examples):
        result = tokenizer(
            text=examples["text"],
            padding="max_length",
            truncation=True,
            max_length=data_config.sequence_length,
        )
        return result

    def load_split(split_name: str, num_examples: int, batch_size: int) -> DataLoader:
        if data_config.dataset_name == "randb":
            dataset = trivial_random_dataset(num_examples, data_config.sequence_length)
        elif data_config.dataset_name == "randb_block":
            dataset = predictable_dataset(num_examples, data_config.sequence_length)
        else:
            dataset = load_dataset(
                path=data_config.dataset_name,
                split=f"{split_name}[:{num_examples}]",
            ).map(
                function=tokenize_function,
                batched=True,
                remove_columns=["text"],
            )

        dataset = dataset.with_format("torch")
        assert isinstance(dataset, dataset_dict.Dataset)

        return DataLoader(
            dataset=dataset,  # type: ignore The dataset is simple enough to be compatible
            batch_size=batch_size,
        )

    train_dataloader = load_split(
        "train", data_config.num_examples_train, data_config.batch_size_train
    )
    test_dataloader = load_split(
        "validation", data_config.num_examples_test, data_config.batch_size_test
    )

    return train_dataloader, test_dataloader


def prepare_training_batch(
    input_ids: torch.Tensor,
    input_embeddings: torch.Tensor,
    token_mask: torch.Tensor,
    block_min_stdev: float,
    block_max_stdev: float,
    bd3_lm_block_size: int,
    distribution: NoiseLevelDistribution,
):
    """Prepare the context needed for a single training step of diffusion
    for a particular block, with noise levels sampled based on a log-normal
    distribution as described in the DiffusionBlocks paper.

    Let
    `B` be the number of sequences in the current batch,
    `S` be the length of the sequence we want to train on, and
    `H` be the size of the embedding.

    Args:
        input_ids: A shape `(B, S)` tensor containing the token ids for a
            batch of sentences.
        input_embeddings: A shape `(B, S, H)` tensor containing the
            embeddings corresponding to `input_ids`.
        token_mask: A shape `(B, S)` tensor containing 1 at locations
            with non-padding tokens, and 0 at locations with padding tokens.
        min_stdev: The minimum noise standard deviation that will be
            assigned to the target block dueing evaluation.
        max_stdev: The maximum noise standard deviation that will be
            assigned to the target block during evaluation.

    Returns:
        A tuple
        `(features, position_ids, stdevs, attention, labels, labels_mask)`.
        - `features` is a shape `(B, (S * 2 - 2), H)` tensor containing both
            clean and noisy embeddings.
        - `position_ids` is a shape `(B, (S * 2 - 2))` tensor containing
            position ids for each embedding.
        - `stdevs` is a shape `(B,)` tensor containing containing noise
            stdevs for each batch.
        - `attention` is a shape `(B, 1, (S * 2 - 2), (S * 2 - 2))` tensor
            containing the attention mask for this batch.
        - `labels is` a shape `(B, (S * 2 - 2))` tensor containing the
            underlying token for each embedding (and thus the denoiser objective).
        - `labels_mask` is a shape `(B, (S * 2 - 2))` tensor containing 1
            iff the token at that position is a non-padding token.
    """
    if not (input_ids.device == input_embeddings.device == token_mask.device):
        raise ValueError(
            f"Expected all tensors to be on the same device. Recieved {input_ids.device=}, {input_embeddings.device=}, {token_mask.device=}."
        )

    num_batches = input_ids.shape[0]
    sequence_length = input_ids.shape[1]
    device = input_embeddings.device

    clean_embeddings: torch.Tensor = input_embeddings[:, :-1, :].clone()
    noisy_embeddings: torch.Tensor = input_embeddings[:, 1:, :].clone()

    # Add noise to noisy embeddings
    stdevs = distribution.get_training_stdevs(
        num_batches, block_min_stdev, block_max_stdev, input_embeddings.device
    )
    noise = torch.randn_like(noisy_embeddings)
    noisy_embeddings += stdevs[:, None, None] * noise

    features = torch.cat([clean_embeddings, noisy_embeddings], dim=1)

    attention = build_bd3lm_train_attention_mask(token_mask, bd3_lm_block_size)

    position_ids = torch.cat(
        [
            torch.arange(0, sequence_length - 1, device=device),
            torch.arange(1, sequence_length, device=device),
        ]
    )
    position_ids = position_ids.broadcast_to((num_batches, sequence_length * 2 - 2))

    labels = torch.cat(
        [
            input_ids[:, :-1],
            input_ids[:, 1:],
        ],
        dim=1,
    )

    labels_mask = torch.cat(
        [
            torch.zeros_like(token_mask[:, :-1]),
            token_mask[:, 1:],
        ],
        dim=1,
    )

    return features, position_ids, stdevs, attention, labels, labels_mask

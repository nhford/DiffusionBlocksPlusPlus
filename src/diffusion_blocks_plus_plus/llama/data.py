import os

import datasets
import lightning as L
import torch
from datasets import load_dataset
from torch.utils.data import DataLoader

from .attention_masks import build_train_attention_mask
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


class TextDataModule(L.LightningDataModule):
    data_name = None
    dataset_kwargs = {}

    def __init__(
        self,
        batch_size: int = 256,
        eval_batch_size: int | None = None,
        num_workers: int | None = None,
    ):
        super().__init__()
        self.batch_size = batch_size
        self.eval_batch_size = eval_batch_size or batch_size
        self.num_workers = num_workers if num_workers is not None else os.cpu_count()
        self.collate_fn = None
        self.datasets = {}
        self.train_key = "train"
        self.val_key = "validation"
        self.test_key = "test"

    def prepare_data(self):
        load_dataset(self.data_name, num_proc=os.cpu_count() // 2)

    def setup(self, stage=None):
        data = load_dataset(self.data_name)
        train_data = data[self.train_key]
        self.datasets["train"] = train_data
        if self.val_key is not None:
            val_data = data[self.val_key]
            self.datasets["val"] = val_data
            self.val_dataloader = self._val_dataloader
        if self.test_key is not None:
            test_data = data[self.test_key]
            self.datasets["test"] = test_data
            self.test_dataloader = self._test_dataloader

    def train_dataloader(self):
        return DataLoader(
            self.datasets["train"],
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True,
            shuffle=True,
        )

    def _val_dataloader(self):
        return DataLoader(
            self.datasets["val"],
            batch_size=self.eval_batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True,
            shuffle=False,
        )

    def _test_dataloader(self):
        return DataLoader(
            self.datasets["test"],
            batch_size=self.eval_batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True,
            shuffle=False,
        )


class TinyStoriesTextDataModule(TextDataModule):
    data_name = "roneneldan/TinyStories"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.train_key = "train"
        self.test_key = "validation"
        self.val_key = "validation"


def load_data(args):
    if args.data_name == "tinystories":
        return TinyStoriesTextDataModule()
    else:
        raise ValueError(f"Invalid data name: {args.data_name}")


def prepare_training_batch(
    input_ids: torch.Tensor,
    input_embeddings: torch.Tensor,
    token_mask: torch.Tensor,
    block_min_stdev: float,
    block_max_stdev: float,
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

    noisy_embeddings /= torch.sqrt(stdevs[:, None, None] ** 2 + 1)

    features = torch.cat([clean_embeddings, noisy_embeddings], dim=1)

    attention = build_train_attention_mask(token_mask)

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

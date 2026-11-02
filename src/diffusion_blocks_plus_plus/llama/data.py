import math
import os

import lightning as L
import torch
from datasets import load_dataset
from scipy.stats import norm
from torch.utils.data import DataLoader


class TextDataModule(L.LightningDataModule):
    data_name = None
    dataset_kwargs = {}

    def __init__(
        self,
        batch_size: int = 2,  # TODO: This does not reflect the paper, and is a lower value due to hardware limitations
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


class DiffusionDataGenerator:
    def __init__(self, p_mean: float, p_std: float, gamma: float):
        self.p_mean = p_mean
        self.p_std = p_std
        self.gamma = gamma

    def get_sigmas(self, num_batches: int, sigma_min: float, sigma_max: float, device):
        """Get a batch of randomly sampled noise standard deviations for a block
        during training.

        Args:
            num_batches: The number of samples to take.
            sigma_min: The minimum noise level assigned to the block at test
                time.
            sigma_max: The maximum noise level assigned to the block at test
                time.
            device: The device to place the output in.

        Returns:
            A tensor of shape `(num_batches,)` with the sampled noise levels.

        Adapted from the DiffusionBlocks repository.
        """
        log_sigma_min = math.log(sigma_min)
        log_sigma_max = math.log(sigma_max)

        log_sigma_range = log_sigma_max - log_sigma_min

        # Expand the range as per the DiffusionBlocks paper.
        log_sigma_min -= log_sigma_range * self.gamma
        log_sigma_max += log_sigma_range * self.gamma

        z_min = (log_sigma_min - self.p_mean) / self.p_std
        z_max = (log_sigma_min - self.p_mean) / self.p_std

        percentile_min = norm.cdf(z_min).item()
        percentile_max = norm.cdf(z_max).item()

        percentile = (
            torch.rand((num_batches,), device=device)
            * (percentile_max - percentile_min)
            + percentile_min
        )

        z = torch.erfinv(percentile)
        log_sigma = self.p_mean + self.p_std * z
        sigma = log_sigma.exp()
        return sigma

    @staticmethod
    def build_train_attention_mask(token_mask: torch.Tensor):
        """Build the attention mask for auto-regressive DiffusionBlocks
        training.

        Throughout the rest of this documentation, let
        `B` be the number of sequences in the current batch, and
        `S` be the length of the token sequence being processed.

        This function builds a rank-4 attention mask that allows tokens to
        attend to themselves and unmasked tokens with lower indices, where 0
        represents attention values that should be allowed, and negative
        infinity represents attention values that are masked out.

        For more information on rank-4 attention masks, see
        https://github.com/huggingface/transformers/pull/27539

        Args:
            token_mask: A shape `(B, S)` tensor containing 1 at locations
                with non-padding tokens, and 0 at locations with padding tokens.


        Returns:
            A tensor of shape `(B, 1, (S * 2 - 2), (S * 2 - 2))` with the
            attention mask.

        Example:
            Let `i` represent negative infinity, if our token_mask is
            `[1,1,1,1,1]`, then this matrix will be returned:

            ```
            0iiiiiii
            00iiiiii
            000iiiii
            0000iiii
            0iii0iii
            00iii0ii
            000iii0i
            0000iii0
            ```
        """
        if len(token_mask.shape) != 2:
            raise ValueError(f"Invalid token mask shape: {token_mask.shape=}")

        num_batches, seq_len = token_mask.shape
        device = token_mask.device
        token_mask = token_mask.bool()

        position_ids = torch.cat(
            [
                torch.arange(0, seq_len - 1, device=device),
                torch.arange(0, seq_len - 1, device=device),
            ]
        )

        is_clean = torch.cat(
            [
                torch.ones(seq_len - 1, dtype=torch.bool, device=device),
                torch.zeros(seq_len - 1, dtype=torch.bool, device=device),
            ]
        )

        query_idx = position_ids[None, None, :, None]
        key_idx = position_ids[None, None, None, :]

        # Attention to previous clean tokens is legal.
        attention_to_clean = (key_idx < query_idx) & is_clean[None, None, None, :]

        # We only want to attend when both the query and key are non-padding tokens.
        is_unmasked = token_mask[:, None, :, None] & token_mask[:, None, None, :]

        # We can always attend to ourselves.
        own_attention = torch.eye(seq_len * 2 - 2, dtype=torch.bool, device=device)

        allowed_attentions = attention_to_clean & is_unmasked | own_attention

        return torch.where(allowed_attentions, 0, float("-inf"))

    def prepare_training_batch(
        self,
        input_ids: torch.Tensor,
        features: torch.Tensor,
        token_mask: torch.Tensor,
        min_stdev: float,
        max_stdev: float,
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
            `(features, position_ids, sigmas, attention, labels, labels_mask)`.
            - `features` is a shape `(B, (S * 2 - 2), H)` tensor containing both
                clean and noisy embeddings.
            - `position_ids` is a shape `(B, (S * 2 - 2))` tensor containing
                position ids for each embedding.
            - `sigmas` is a shape `(B,)` tensor containing containing noise
                stdevs for each batch.
            - `attention` is a shape `(B, 1, (S * 2 - 2), (S * 2 - 2))` tensor
                containing the attention mask for this batch.
            - `labels is` a shape `(B, (S * 2 - 2))` tensor containing the
                underlying token for each embedding (and thus the denoiser objective).
            - `labels_mask` is a shape `(B, (S * 2 - 2))` tensor containing 1
                iff the token at that position is a non-padding token.
        """
        if not (input_ids.device == features.device == token_mask.device):
            raise ValueError(
                f"Expected all tensors to be on the same device. Recieved {input_ids.device=}, {features.device=}, {token_mask.device=}."
            )

        num_batches = input_ids.shape[0]
        sequence_length = input_ids.shape[1]
        device = features.device

        clean_embeddings: torch.Tensor = features[:, :-1, :]
        noisy_embeddings: torch.Tensor = features[:, 1:, :]

        # Add noise to noisy embeddings
        sigmas = self.get_sigmas(num_batches, min_stdev, max_stdev, features.device)
        noise = torch.randn_like(noisy_embeddings)
        noisy_embeddings += sigmas[:, None, None] * noise

        features = torch.cat([clean_embeddings, noisy_embeddings], dim=1)

        attention = self.build_train_attention_mask(token_mask)

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
                token_mask[:, :-1],
                token_mask[:, 1:],
            ],
            dim=1,
        )

        return features, position_ids, sigmas, attention, labels, labels_mask

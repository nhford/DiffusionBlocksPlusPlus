import os
from datetime import datetime

import pandas
import torch
import torch.nn.functional as F
from datasets import dataset_dict, load_dataset
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import (
    AutoTokenizer,
    LlamaConfig,
    LlamaForCausalLM,
    LlamaTokenizerFast,
)
from transformers.modeling_outputs import CausalLMOutputWithPast

from .llama.model import LlamaDiTConfig, LlamaDiTForCausalLM
from .plot_data import generate_plot

SEQUENCE_LENGTH = 20  # TODO: 256 is prod
BATCH_SIZE = 10  # TODO: Should be 256 as in original paper
LEARNING_RATE = 0.0001
USE_DBLOCK = True
DEBUG_LEARN_NOISE = True
DEBUG_NOISE_MIN = 0
DEBUG_NOISE_MAX = 1
DEBUG_SMALL_DATASET = True


def build_attention_mask(attention_mask: torch.Tensor):
    """
    Build the attention mask for auto-regressive diffusion-blocks training.

    This function builds a
    [rank-4 attention mask](https://github.com/huggingface/transformers/pull/27539)
    tensor that can be used for causal diffusion language models.
    [Note](https://github.com/huggingface/transformers/pull/27539#issuecomment-2531760367)
    that negative infinity should be used for attention weights that we want to
    mask out, and 0 should be used for attention weights we want to keep.


    Args:
        attention_mask: A TODO


    Returns:
        A tensor of shape (1, 1, seq_length * 2, seq_length * 2) with the
        attention mask.
        The shape for the attention mask is supposed to be (BxHxSxS), or (batch,
        head, query, key).

    Example:
        Let i represent negative infinity, if our max_seq_length is 4, then this matrix will be returned:

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
    if len(attention_mask.shape) != 2:
        raise ValueError(f"Invalid attention mask shape: {attention_mask.shape=}")

    num_batches, seq_len = attention_mask.shape

    result = torch.full((num_batches, 1, seq_len * 2, seq_len * 2), float("-inf"))

    for batch in range(num_batches):
        for query_idx in range(seq_len):
            # All tokens (clean and noisy) can attend to themselves
            result[batch, 0, query_idx, query_idx] = 0
            result[batch, 0, query_idx + seq_len, query_idx + seq_len] = 0
            for key_idx in range(query_idx):
                if attention_mask[batch, key_idx] == 1:
                    # tokens can attend to previous, valid, clean tokens.
                    result[batch, 0, query_idx, key_idx] = 0
                    result[batch, 0, query_idx + seq_len, key_idx] = 0

    return result


if __name__ == "__main__":
    # See Appendix E.4 in the Sakana.ai paper
    if USE_DBLOCK:
        config = LlamaDiTConfig(
            vocab_size=32000,
            num_hidden_layers=12,
            hidden_size=768,
            num_attention_heads=12,
            intermediate_size=3072,
        )

        model = LlamaDiTForCausalLM(config)
    else:
        config = LlamaConfig(
            vocab_size=32000,
            num_hidden_layers=12,
            hidden_size=768,
            num_attention_heads=12,
            intermediate_size=3072,
        )

        model = LlamaForCausalLM(config)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"{total_params=}")
    print(f"{trainable_params=}")

    tokenizer = AutoTokenizer.from_pretrained("NousResearch/Llama-2-7b-chat-hf")
    assert isinstance(tokenizer, LlamaTokenizerFast)
    # This is because the attention mask must not attend to padding tokens.
    tokenizer.padding_side = "right"
    tokenizer.add_eos_token = True

    if DEBUG_SMALL_DATASET:
        train_dataset = load_dataset("roneneldan/TinyStories", split="train[:1000]")
    else:
        train_dataset = load_dataset("roneneldan/TinyStories", split="train")
    assert isinstance(train_dataset, dataset_dict.Dataset)
    dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE)  # type: ignore The dataset is simple enough to be compatible

    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)

    results = []
    progress = tqdm(dataloader)
    for batch_idx, batch in enumerate(progress):
        batch_text: list[str] = batch["text"]

        batch_size = len(batch_text)

        tokens = tokenizer(
            batch_text,
            padding="max_length",
            truncation=True,
            max_length=SEQUENCE_LENGTH + 1,
            return_tensors="pt",
        )

        token_ids: torch.Tensor = tokens.input_ids
        assert isinstance(token_ids, torch.Tensor) and token_ids.shape == (
            batch_size,
            SEQUENCE_LENGTH + 1,
        )
        if DEBUG_LEARN_NOISE:
            token_ids = torch.randint_like(token_ids, 2)

        masked_positions: torch.Tensor = tokens.attention_mask
        assert isinstance(
            masked_positions, torch.Tensor
        ) and masked_positions.shape == (
            batch_size,
            SEQUENCE_LENGTH + 1,
        )
        if DEBUG_LEARN_NOISE:
            masked_positions = torch.ones_like(masked_positions)

        embedding_fn = model.model.embed_tokens
        assert isinstance(embedding_fn, nn.Embedding)

        clean_embeddings: torch.Tensor = embedding_fn(token_ids[:, :-1])
        clean_embeddings = F.normalize(clean_embeddings, p=2, dim=-1)
        assert clean_embeddings.shape == (
            batch_size,
            SEQUENCE_LENGTH,
            config.hidden_size,
        )
        noisy_embeddings: torch.Tensor = embedding_fn(token_ids[:, 1:])
        noisy_embeddings = F.normalize(noisy_embeddings, p=2, dim=-1)
        assert noisy_embeddings.shape == (
            batch_size,
            SEQUENCE_LENGTH,
            config.hidden_size,
        )

        # Add noise to noisy embeddings
        # TODO modify noise and base scaling to match the paper
        noise_stdevs = (
            torch.rand((batch_size, 1, 1)) * (DEBUG_NOISE_MAX - DEBUG_NOISE_MIN)
            + DEBUG_NOISE_MIN
        )
        noise = torch.randn_like(noisy_embeddings)
        noisy_embeddings = noise_stdevs * noise + (1 - noise_stdevs) * noisy_embeddings

        input_embeddings = torch.cat([clean_embeddings, noisy_embeddings], dim=1)
        assert input_embeddings.shape == (
            batch_size,
            SEQUENCE_LENGTH * 2,
            config.hidden_size,
        )

        attention_mask = build_attention_mask(masked_positions[:, :-1])
        assert attention_mask.shape == (
            batch_size,
            1,
            SEQUENCE_LENGTH * 2,
            SEQUENCE_LENGTH * 2,
        )

        position_ids = torch.cat(
            [torch.arange(0, SEQUENCE_LENGTH), torch.arange(1, SEQUENCE_LENGTH + 1)]
        )
        position_ids = position_ids.view((1, SEQUENCE_LENGTH * 2))

        optimizer.zero_grad()
        outputs: CausalLMOutputWithPast = model(
            inputs_embeds=input_embeddings,
            attention_mask=attention_mask,
            position_ids=position_ids,
        )
        logits = outputs.logits
        assert logits is not None and logits.shape == (
            batch_size,
            SEQUENCE_LENGTH * 2,
            config.vocab_size,
        )
        denoise_logits = logits[:, SEQUENCE_LENGTH:, :]
        denoise_logits = denoise_logits.transpose(1, 2)
        targets = token_ids[:, 1:]
        loss = F.cross_entropy(denoise_logits, targets, reduction="none")
        loss = loss * masked_positions[:, 1:]
        loss = loss.sum() / masked_positions[:, 1:].sum()
        loss.backward()
        optimizer.step()

        results.append({"batch_idx": batch_idx, "loss": loss.item()})
        progress.set_postfix({"loss": loss.item()})

    name = datetime.now().isoformat()
    os.makedirs("./data", exist_ok=True)
    results = pandas.DataFrame(results)
    results.to_csv(f"./data/{name}.csv")

    generate_plot(name)

    model.save_pretrained(f"./checkpoints/{name}.ckpt")

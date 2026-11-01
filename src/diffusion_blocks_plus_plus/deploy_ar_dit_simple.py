import os
from datetime import datetime

import pandas
import torch
import torch.nn.functional as F
from datasets import dataset_dict, load_dataset
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import (
    AutoTokenizer,
    LlamaTokenizerFast,
)
from transformers.modeling_outputs import CausalLMOutputWithPast

from .llama.model import LlamaDiTConfig, LlamaDiTForCausalLM
from .plot_data import generate_plot

SEQUENCE_LENGTH = 128  # TODO: 256 is prod
BATCH_SIZE = 2  # TODO: Should be 256 as in original paper
LEARNING_RATE = 0.001
EMBED_TIMESTEPS = False

DEBUG_LEARN_NOISE = False
DEBUG_NOISE_MIN = 0.5
DEBUG_NOISE_MAX = 1
DEBUG_SMALL_DATASET = True
DEBUG_FORCE_COND = False
DEBUG_LOG_VRAM = True


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

    result = torch.full(
        (num_batches, 1, seq_len * 2, seq_len * 2),
        float("-inf"),
    )

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

    return result.to(attention_mask.device)


def get_vram_usage():
    return torch.cuda.memory_allocated()


def train_block(model: LlamaDiTForCausalLM, block_idx: int, dataloader: DataLoader):
    for i in range(model.num_blocks):
        model.unload_block(i)
    model.load_block(block_idx)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to_device(device)

    total_params = sum(p.numel() for p in model.loaded_parameters())
    trainable_params = sum(
        p.numel() for p in model.loaded_parameters() if p.requires_grad
    )
    print(f"{total_params=}")
    print(f"{trainable_params=}")

    optimizer = AdamW(model.loaded_parameters(), lr=LEARNING_RATE)

    results = []
    progress = tqdm(dataloader)
    for batch_idx, batch in enumerate(progress):
        vram_usage_dict = {"model": 0, "activations": 0, "gradients": 0, "optimizer": 0}
        if DEBUG_LOG_VRAM:
            for param in model.loaded_parameters():
                vram_usage_dict["model"] += param.numel() * param.element_size()

        batch_size = len(batch["input_ids"])

        if DEBUG_LOG_VRAM:
            vram_usage_dict["activations"] -= get_vram_usage()
        token_ids: torch.Tensor = batch["input_ids"].to(device)
        assert isinstance(token_ids, torch.Tensor) and token_ids.shape == (
            batch_size,
            SEQUENCE_LENGTH + 1,
        )
        if DEBUG_LEARN_NOISE:
            token_ids = torch.randint_like(token_ids, 2)

        masked_positions: torch.Tensor = batch["attention_mask"].to(device)
        assert isinstance(
            masked_positions, torch.Tensor
        ) and masked_positions.shape == (
            batch_size,
            SEQUENCE_LENGTH + 1,
        )
        if DEBUG_LEARN_NOISE:
            masked_positions = torch.ones_like(masked_positions)

        clean_embeddings: torch.Tensor = model.input_embedding(token_ids[:, :-1])
        clean_embeddings = F.normalize(clean_embeddings, p=2, dim=-1)
        assert clean_embeddings.shape == (
            batch_size,
            SEQUENCE_LENGTH,
            config.hidden_size,
        )
        noisy_embeddings: torch.Tensor = model.input_embedding(token_ids[:, 1:])
        noisy_embeddings = F.normalize(noisy_embeddings, p=2, dim=-1)
        assert noisy_embeddings.shape == (
            batch_size,
            SEQUENCE_LENGTH,
            config.hidden_size,
        )

        # Add noise to noisy embeddings
        # TODO modify noise and base scaling to match the paper
        noise_stdevs = (
            torch.rand((batch_size, 1, 1), device=device)
            * (DEBUG_NOISE_MAX - DEBUG_NOISE_MIN)
            + DEBUG_NOISE_MIN
        )
        noise = torch.randn_like(noisy_embeddings)
        noisy_embeddings = noise_stdevs * noise + (1 - noise_stdevs) * noisy_embeddings

        if DEBUG_FORCE_COND:
            # A Debug shim we can put in place is to rerandomize the noise conditioning and perturb outputs to add a pure noise value
            noise_stdevs = torch.rand((batch_size, 1, 1), device=device)

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
        ).to(device)
        position_ids = position_ids.view((1, SEQUENCE_LENGTH * 2))

        # TODO: Move the responsibility of block selection into a method of model.
        current_block = model.blocks_loaded[block_idx]
        assert current_block is not None

        outputs: CausalLMOutputWithPast = current_block(
            inputs_embeds=input_embeddings,
            attention_mask=attention_mask,
            position_ids=position_ids,
            **(
                dict(timestep=noise_stdevs.view((batch_size,)))
                if EMBED_TIMESTEPS
                else {}
            ),
        )
        if DEBUG_LOG_VRAM:
            vram_usage_dict["activations"] += get_vram_usage()

        logits = outputs.logits
        assert logits is not None and logits.shape == (
            batch_size,
            SEQUENCE_LENGTH * 2,
            config.vocab_size,
        )
        if DEBUG_FORCE_COND:
            logits[:, :, :1] += (noise_stdevs - 0.5) * 8
        denoise_logits = logits[:, SEQUENCE_LENGTH:, :]
        denoise_logits = denoise_logits.transpose(1, 2)
        targets = token_ids[:, 1:]
        loss = F.cross_entropy(denoise_logits, targets, reduction="none")
        loss = loss * masked_positions[:, 1:]
        loss = loss.sum() / masked_positions[:, 1:].sum()

        loss.backward()

        if DEBUG_LOG_VRAM:
            for param in model.loaded_parameters():
                grad = param.grad
                if grad is not None:
                    vram_usage_dict["gradients"] += grad.numel() * grad.element_size()

        optimizer.step()

        if DEBUG_LOG_VRAM:
            for optimizer_params in optimizer.state.values():
                if not isinstance(optimizer_params, dict):
                    continue
                for state_entry in optimizer_params.values():
                    vram_usage_dict["optimizer"] += (
                        state_entry.numel() * state_entry.element_size()
                    )

        optimizer.zero_grad()

        vram_usage_dict = {
            **vram_usage_dict,
            "total": sum(vram_usage_dict.values()),
        }
        vram_usage_dict = {"vram_" + k: v / 1e9 for k, v in vram_usage_dict.items()}
        results.append({"batch_idx": batch_idx, "loss": loss.item(), **vram_usage_dict})
        progress.set_postfix({"loss": loss.item(), **vram_usage_dict})
    return results


if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained("NousResearch/Llama-2-7b-chat-hf")
    assert isinstance(tokenizer, LlamaTokenizerFast)
    # This is because the attention mask must not attend to padding tokens.
    tokenizer.padding_side = "right"
    tokenizer.add_eos_token = True

    def tokenize_function(examples):
        result = tokenizer(
            examples["text"],
            padding="max_length",
            truncation=True,
            max_length=SEQUENCE_LENGTH + 1,
        )
        return result

    if DEBUG_SMALL_DATASET:
        train_dataset = load_dataset("roneneldan/TinyStories", split="train[:1000]")
    else:
        train_dataset = load_dataset("roneneldan/TinyStories", split="train")
    assert isinstance(train_dataset, dataset_dict.Dataset)

    train_dataset = train_dataset.map(
        tokenize_function, batched=True, remove_columns=["text"]
    )
    train_dataset = train_dataset.with_format("torch")
    dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE)  # type: ignore The dataset is simple enough to be compatible

    # See Appendix E.4 in the Sakana.ai paper
    config = LlamaDiTConfig(
        vocab_size=32000,
        num_hidden_layers=48,
        hidden_size=768,
        num_attention_heads=12,
        intermediate_size=3072,
        num_blocks=4,
        use_timestep_embedding=EMBED_TIMESTEPS,
    )

    model = LlamaDiTForCausalLM(config)

    for i in range(model.num_blocks):
        results = train_block(model, i, dataloader)

        name = datetime.now().isoformat() + "_" + str(i)
        os.makedirs("./data", exist_ok=True)
        results = pandas.DataFrame(results)
        results.to_csv(f"./data/{name}.csv")

        generate_plot(name)

        model.save_checkpoint(f"./checkpoints/{name}")

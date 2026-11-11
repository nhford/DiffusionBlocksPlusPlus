import argparse
import math
import os
from datetime import datetime

import pandas
import torch
import torch.nn.functional as F
from datasets import dataset_dict, load_dataset
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer, LlamaTokenizerFast
from transformers.modeling_outputs import CausalLMOutputWithPast

from diffusion_blocks_plus_plus.llama.attention_masks import (
    build_train_attention_mask,
    mask_to_pos,
)
from diffusion_blocks_plus_plus.llama.data import (
    predictable_dataset,
    prepare_training_batch,
    trivial_random_dataset,
)
from diffusion_blocks_plus_plus.llama.noise_level_manager import NoiseLevelDistribution
from diffusion_blocks_plus_plus.llama.vram_tracker import MemoryTracker

from .llama.model import LlamaDiTConfig, LlamaDiTForCausalLM
from .plot_data import generate_plot

SEQUENCE_LENGTH = 128  # TODO: 256 is prod
BATCH_SIZE = 2  # TODO: Should be 256 as in original paper
LEARNING_RATE = 0.001
EMBED_TIMESTEPS = False
EVAL_NUM_STEPS = 10
NUM_EPOCHS = 1

DEBUG_LEARN_NOISE = True
DEBUG_DATASET_CAP = 100
DEBUG_FORCE_COND = False
DEBUG_LOG_VRAM = True


def train_block(
    model: LlamaDiTForCausalLM,
    block_idx: int,
    dataloader: DataLoader,
    device: torch.device,
    noise_schedule: NoiseLevelDistribution,
):
    for i in range(model.num_blocks):
        model.unload_block(i)
    model.load_block(block_idx)
    current_block = model.blocks_loaded[block_idx]
    assert current_block is not None
    current_block.train()
    model.input_embedding.train()

    min_stdev, max_stdev = model.noise_boundaries(block_idx)

    model.to_device(device)

    optimizer = AdamW(model.loaded_parameters(), lr=LEARNING_RATE)

    memory_tracker = MemoryTracker(model, optimizer, device)

    results = []

    progress = tqdm(dataloader, leave=False, desc="Batch")
    for batch_idx, batch in enumerate(progress):
        memory_tracker.reset()
        memory_tracker.measure_mem_weights()
        memory_tracker.begin_forward()

        token_ids = batch["input_ids"].to(device)
        assert isinstance(token_ids, torch.Tensor)
        # assert token_ids.shape == (BATCH_SIZE, SEQUENCE_LENGTH + 1)

        masked_positions = batch["attention_mask"].to(device)
        assert isinstance(masked_positions, torch.Tensor)
        # assert masked_positions.shape == (BATCH_SIZE, SEQUENCE_LENGTH + 1)

        features, position_ids, sigmas, attention, labels, labels_mask = (
            prepare_training_batch(
                input_ids=token_ids,
                input_embeddings=model.embed_input_ids(token_ids),
                token_mask=masked_positions,
                block_min_stdev=min_stdev,
                block_max_stdev=max_stdev,
                distribution=noise_schedule,
            )
        )

        outputs: CausalLMOutputWithPast = current_block(
            inputs_embeds=features,
            attention_mask=attention,
            position_ids=position_ids,
            **(dict(timestep=sigmas.log()) if EMBED_TIMESTEPS else {}),
        )

        memory_tracker.measure_mem_act()

        logits = outputs.logits
        assert isinstance(logits, torch.Tensor)
        # assert logits.shape == (BATCH_SIZE, SEQUENCE_LENGTH * 2, config.vocab_size)

        loss = F.cross_entropy(logits.permute(0, 2, 1), labels, reduction="none")
        loss = (loss * labels_mask).sum() / labels_mask.sum()
        loss.backward()

        memory_tracker.measure_mem_grad()

        optimizer.step()

        memory_tracker.measure_mem_optim()

        optimizer.zero_grad()

        results.append(
            {"batch_idx": batch_idx, "loss": loss.item(), **memory_tracker.as_dict()}
        )
        progress.set_postfix({"loss": loss.item()})
    return results


def train_model(
    model: LlamaDiTForCausalLM,
    dataloader: DataLoader,
    device: torch.device,
    noise_schedule: NoiseLevelDistribution,
) -> list[dict]:
    print(f"Training model for {NUM_EPOCHS} epochs and {len(dataloader.dataset)} samples.")  # type: ignore[reportArgumentType]

    results = []

    for block_idx in tqdm(range(model.num_blocks - 1, -1, -1), desc="Block"):
        for epoch_idx in tqdm(range(NUM_EPOCHS), leave=False, desc="Epoch"):
            block_results = train_block(
                model, block_idx, dataloader, device, noise_schedule
            )
            results += [{"block_idx": block_idx, **row} for row in block_results]

    return results


@torch.no_grad
def evaluate_model(
    model: LlamaDiTForCausalLM,
    dataloader: DataLoader,
    device: torch.device,
    noise_schedule: NoiseLevelDistribution,
):
    """Predict every token in a dataset using NTP and return the accuracy."""

    model.input_embedding.to(device)
    # We know the dataset has a length
    print(f"Evaluating model on {len(dataloader.dataset)} samples.")  # type: ignore[reportArgumentType]

    num_examples = 0
    num_correct = 0

    progress = tqdm(dataloader, desc="Batch")
    for batch_idx, batch in enumerate(progress):
        token_ids = batch["input_ids"].to(device)
        assert isinstance(token_ids, torch.Tensor)
        # assert token_ids.shape == (BATCH_SIZE, SEQUENCE_LENGTH + 1)

        masked_positions = batch["attention_mask"].to(device)
        assert isinstance(masked_positions, torch.Tensor)
        # assert masked_positions.shape == (BATCH_SIZE, SEQUENCE_LENGTH + 1)

        # Add 1 to the per-channel variance due to the embedding variance.
        initial_variance = model.block_boundaries[-1] ** 2 + 1
        initial_stdev = math.sqrt(initial_variance)

        clean_embeddings = model.embed_input_ids(token_ids[:, :-1])
        # assert clean_embeddings.shape == (
        #     BATCH_SIZE,
        #     SEQUENCE_LENGTH,
        #     clean_embeddings.shape[2],
        # )

        attention_mask = build_train_attention_mask(masked_positions)
        # assert attention_mask.shape == (
        #     BATCH_SIZE,
        #     1,
        #     SEQUENCE_LENGTH * 2,
        #     SEQUENCE_LENGTH * 2,
        # )

        position_ids = mask_to_pos(masked_positions)
        position_ids = torch.cat([position_ids[:, :-1], position_ids[:, 1:]], dim=1)
        # assert position_ids.shape == (BATCH_SIZE, SEQUENCE_LENGTH * 2)

        def get_logits(noisy_embeddings: torch.Tensor, timestep: float) -> torch.Tensor:
            """Get the logit predictions for an embedding's denoised embedding."""
            current_block_idx = model.get_block_from_timestep(timestep)

            for block_idx in range(model.num_blocks):
                if block_idx != current_block_idx:
                    model.unload_block(block_idx)
            model.load_block(current_block_idx)
            current_block = model.blocks_loaded[current_block_idx]
            assert current_block is not None
            current_block.to(device)
            current_block.eval()
            model.input_embedding.eval()

            timestep_embedding_kwarg = {}
            if EMBED_TIMESTEPS:
                timestep_embedding_kwarg["timestep"] = torch.full(
                    (noisy_embeddings.shape[0],), timestep
                )

            inputs_embeds = torch.cat([clean_embeddings, noisy_embeddings], dim=1)

            outputs: CausalLMOutputWithPast = current_block(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                position_ids=position_ids,
                **timestep_embedding_kwarg,
            )

            logits = outputs.logits
            assert logits is not None
            return logits[:, SEQUENCE_LENGTH:, :]

        def ODE(noisy_embeddings: torch.Tensor, timestep: float) -> torch.Tensor:
            """Get the change in hidden embedding with respect to timestep"""
            noisy_logits = get_logits(noisy_embeddings, timestep)
            expected = model.embed_output_logits(noisy_logits)
            # Our model is that the Gaussians scale at x=e^t, so the time derivative dx/dt is x=e^t.
            return noisy_embeddings - expected

        noise_levels = noise_schedule.get_discrete_stdevs(EVAL_NUM_STEPS)

        noisy_embeddings = torch.randn_like(clean_embeddings) * initial_stdev
        for stdev, next_stdev in tqdm(
            list(zip(noise_levels[:-1], noise_levels[1:])),
            desc="Denoise Step",
            leave=False,
        ):
            # TODO: Implement alternative samplers
            dt = math.log(next_stdev) - math.log(stdev)
            noisy_embeddings += ODE(noisy_embeddings, math.log(stdev)) * dt
        t_final = math.log(noise_levels[-1])
        predictions = get_logits(noisy_embeddings, t_final).argmax(-1)
        # assert predictions.shape == (BATCH_SIZE, SEQUENCE_LENGTH)
        correct = predictions == token_ids[:, 1:]
        prediction_targets = masked_positions[:, 1:] & masked_positions[:, :-1]
        correct = correct & prediction_targets

        num_correct += correct.sum().item()
        num_examples += prediction_targets.sum().item()

        progress.set_postfix({"acc": num_correct / num_examples})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="ProgramName",
        description="What the program does",
        epilog="Text at the bottom of help",
    )

    parser.add_argument("action")
    parser.add_argument("--from-ckpt")
    parser.add_argument("-c", "--to-ckpt")
    parser.add_argument("-d", "--data-path")

    args = parser.parse_args()

    if args.action not in ["train", "evaluate"]:
        raise ValueError("Must choose train or evaluate")
    action = args.action

    timestamp = datetime.now().isoformat()
    checkpoint_save_path = f"./checkpoints/ckpt-{timestamp}"
    if args.to_ckpt is not None:
        checkpoint_save_path = args.to_ckpt

    checkpoint_load_path = args.from_ckpt

    metrics_save_path = f"./data/{timestamp}.csv"
    if args.data_path is not None:
        metrics_save_path = args.data_path

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

    if DEBUG_LEARN_NOISE:
        train_dataset = predictable_dataset(DEBUG_DATASET_CAP, SEQUENCE_LENGTH + 1)
        test_dataset = predictable_dataset(DEBUG_DATASET_CAP // 10, SEQUENCE_LENGTH + 1)
    else:
        train_split = (
            "train" if DEBUG_DATASET_CAP is None else f"train[:{DEBUG_DATASET_CAP}]"
        )
        train_dataset = load_dataset("roneneldan/TinyStories", split=train_split)
        train_dataset = train_dataset.map(
            tokenize_function, batched=True, remove_columns=["text"]
        )
        test_split = (
            "valid"
            if DEBUG_DATASET_CAP is None
            else f"valid[:{DEBUG_DATASET_CAP // 10}]"
        )
        test_dataset = load_dataset("roneneldan/TinyStories", split=test_split)
        test_dataset = test_dataset.map(
            tokenize_function, batched=True, remove_columns=["text"]
        )
    train_dataset = train_dataset.with_format("torch")
    test_dataset = test_dataset.with_format("torch")
    assert isinstance(train_dataset, dataset_dict.Dataset)
    assert isinstance(test_dataset, dataset_dict.Dataset)

    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE)  # type: ignore The dataset is simple enough to be compatible
    test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE)  # type: ignore The dataset is simple enough to be compatible

    noise_schedule = NoiseLevelDistribution()

    # See Appendix E.4 in the Sakana.ai paper
    config = LlamaDiTConfig(
        vocab_size=32000,
        num_hidden_layers=48,
        hidden_size=768,
        num_attention_heads=12,
        intermediate_size=3072,
        num_blocks=4,
        use_timestep_embedding=EMBED_TIMESTEPS,
        block_boundaries=noise_schedule.get_block_stdevs(4),
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = LlamaDiTForCausalLM(config)
    if checkpoint_load_path is not None:
        model.load_checkpoint(checkpoint_load_path)

    if action == "train":
        results_dict = train_model(model, train_dataloader, device, noise_schedule)

        model.save_checkpoint(checkpoint_save_path)
        pandas.DataFrame(results_dict).to_csv(metrics_save_path)

    if action == "evaluate":
        evaluate_model(model, test_dataloader, device, noise_schedule)

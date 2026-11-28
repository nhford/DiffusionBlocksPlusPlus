import argparse
import math
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import pandas
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR, LRScheduler
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers.modeling_outputs import CausalLMOutputWithPast

import wandb

from .llama.attention_masks import (
    build_bd3lm_train_attention_mask,
    mask_to_pos,
)
from .llama.config import (
    LlamaDiTRunConfig,
    LlamaTestConfig,
    LlamaTrainConfig,
    load_config,
    save_config,
)
from .llama.data import (
    build_timesteps,
    load_data,
    prepare_training_batch,
)
from .llama.model import LlamaDiTForCausalLM
from .llama.noise_level_manager import NoiseLevelDistribution
from .llama.samplers import sample
from .llama.vram_tracker import MemoryTracker


def train_block(
    model: LlamaDiTForCausalLM,
    block_idx: int,
    dataloader: DataLoader,
    device: torch.device,
    noise_schedule: NoiseLevelDistribution,
    train_config: LlamaTrainConfig,
    optimizer: AdamW,
    scheduler: LRScheduler,
    step_offset: int,
):
    model.train()

    min_stdev, max_stdev = (
        model.block_boundaries[block_idx],
        model.block_boundaries[block_idx + 1],
    )

    memory_tracker = MemoryTracker(model, optimizer, device)

    results = []

    progress = tqdm(dataloader, leave=False, desc="Batch")
    for batch_idx, batch in enumerate(progress):
        batch_size, sequence_length = batch["input_ids"].shape

        memory_tracker.reset()
        memory_tracker.measure_mem_weights(block_idx)
        memory_tracker.begin_forward()

        token_ids = batch["input_ids"].to(device)
        assert isinstance(token_ids, torch.Tensor)
        assert token_ids.shape == (batch_size, sequence_length)

        masked_positions = batch["attention_mask"].to(device)
        assert isinstance(masked_positions, torch.Tensor)
        assert masked_positions.shape == (batch_size, sequence_length)

        features, position_ids, sigmas, attention, labels, labels_mask = (
            prepare_training_batch(
                input_ids=token_ids,
                input_embeddings=model.embed_input_ids(token_ids),
                token_mask=masked_positions,
                block_min_stdev=min_stdev,
                block_max_stdev=max_stdev,
                distribution=noise_schedule,
                bd3_lm_block_size=train_config.bd3_lm_block_size,
            )
        )

        outputs: CausalLMOutputWithPast = model(
            inputs_embeds=features,
            attention_mask=attention,
            position_ids=position_ids,
            timestep=build_timesteps(
                features, sigmas.log(), model.timestep_conditioning
            ),
            block_idx=block_idx,
        )

        memory_tracker.measure_mem_act()

        logits = outputs.logits
        assert isinstance(logits, torch.Tensor)
        assert logits.shape == (batch_size, sequence_length * 2 - 2, model.vocab_size)

        loss = F.cross_entropy(logits.permute(0, 2, 1), labels, reduction="none")
        loss = (loss * labels_mask).sum() / labels_mask.sum()
        loss.backward()

        memory_tracker.measure_mem_grad(block_idx)

        current_lr = optimizer.param_groups[0]["lr"]
        optimizer.step()
        scheduler.step()

        memory_tracker.measure_mem_optim()

        optimizer.zero_grad()

        row = {"batch_idx": batch_idx, "loss": loss.item(), **memory_tracker.as_dict()}
        results.append(row)
        if wandb.run is not None:
            prefix = f"train/block_{block_idx}"
            wandb.log(
                {
                    f"{prefix}/step": step_offset + batch_idx,
                    f"{prefix}/loss": row["loss"],
                    f"{prefix}/lr": current_lr,
                    **{
                        f"vmem/{prefix}/{k}": v
                        for k, v in row.items()
                        if k.startswith("mem_")
                    },
                }
            )
        progress.set_postfix({"loss": loss.item()})
    return results


def train_model(
    model: LlamaDiTForCausalLM,
    dataloader: DataLoader,
    device: torch.device,
    noise_schedule: NoiseLevelDistribution,
    train_config: LlamaTrainConfig,
) -> list[dict]:
    print(
        f"Training model on {len(dataloader.dataset)} samples for {train_config.num_epochs} epochs."  # type: ignore[reportArgumentType]
    )

    results = []

    if wandb.run is not None:
        for block_idx in range(model.num_blocks):
            # Each block gets its own charts, plotted against its own step.
            wandb.define_metric(f"train/block_{block_idx}/step")
            wandb.define_metric(
                f"train/block_{block_idx}/*",
                step_metric=f"train/block_{block_idx}/step",
            )

    for block_idx in tqdm(range(model.num_blocks - 1, -1, -1), desc="Block"):
        model.set_primary_device(device)
        model.set_loaded_block(block_idx)

        optimizer = AdamW(
            model.block_parameters(block_idx, True), lr=train_config.learning_rate
        )

        scheduler = LambdaLR(
            optimizer=optimizer,
            lr_lambda=lambda step: min((step + 1) / train_config.warmup_steps, 1.0),
        )

        for epoch_idx in tqdm(
            range(train_config.num_epochs), leave=False, desc="Epoch"
        ):
            block_results = train_block(
                model=model,
                block_idx=block_idx,
                dataloader=dataloader,
                device=device,
                noise_schedule=noise_schedule,
                train_config=train_config,
                optimizer=optimizer,
                scheduler=scheduler,
                step_offset=epoch_idx * len(dataloader),
            )
            results += [{"block_idx": block_idx, **row} for row in block_results]

    return results


@torch.no_grad
def evaluate_model(
    model: LlamaDiTForCausalLM,
    dataloader: DataLoader,
    device: torch.device,
    noise_schedule: NoiseLevelDistribution,
    test_config: LlamaTestConfig,
):
    """Predict every token in a dataset using NTP and return the accuracy."""
    model.set_primary_device(device)

    # We know the dataset has a length
    print(f"Evaluating model on {len(dataloader.dataset)} samples.")  # type: ignore[reportArgumentType]

    num_examples = 0
    num_correct = 0

    progress = tqdm(dataloader, desc="Batch")
    for batch_idx, batch in enumerate(progress):
        batch_size, sequence_length = batch["input_ids"].shape

        token_ids = batch["input_ids"].to(device)
        assert isinstance(token_ids, torch.Tensor)
        assert token_ids.shape == (batch_size, sequence_length)

        masked_positions = batch["attention_mask"].to(device)
        assert isinstance(masked_positions, torch.Tensor)
        assert masked_positions.shape == (batch_size, sequence_length)

        # Add 1 to the per-channel variance due to the embedding variance.
        initial_variance = model.block_boundaries[-1] ** 2 + 1
        initial_stdev = math.sqrt(initial_variance)

        clean_embeddings = model.embed_input_ids(token_ids[:, :-1])
        assert clean_embeddings.shape == (
            batch_size,
            sequence_length - 1,
            clean_embeddings.shape[2],
        )

        attention_mask = build_bd3lm_train_attention_mask(
            masked_positions, test_config.bd3_lm_block_size
        )
        assert attention_mask.shape == (
            batch_size,
            1,
            sequence_length * 2 - 2,
            sequence_length * 2 - 2,
        )

        position_ids = mask_to_pos(masked_positions)
        position_ids = torch.cat([position_ids[:, :-1], position_ids[:, 1:]], dim=1)
        assert position_ids.shape == (batch_size, (sequence_length - 1) * 2)

        @torch.no_grad
        def get_logits(
            noisy_embeddings: torch.Tensor,
            timestep: float,
            *,
            clean_embeddings=clean_embeddings,
            attention_mask=attention_mask,
            position_ids=position_ids,
            sequence_length=sequence_length,
        ) -> torch.Tensor:
            """Get the logit predictions for an embedding's denoised embedding."""
            model.eval()

            # Normalize inputs
            noisy_embeddings = noisy_embeddings / math.sqrt(math.exp(timestep) ** 2 + 1)

            inputs_embeds = torch.cat([clean_embeddings, noisy_embeddings], dim=1)

            timestep_tensor = build_timesteps(
                inputs_embeds,
                timestep,
                model.timestep_conditioning,
            )

            outputs: CausalLMOutputWithPast = model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                position_ids=position_ids,
                timestep=timestep_tensor,
            )

            logits = outputs.logits
            assert logits is not None
            return logits[:, sequence_length - 1 :, :]

        @torch.no_grad
        def ODE(noisy_embeddings: torch.Tensor, timestep: float) -> torch.Tensor:
            """Get the change in hidden embedding with respect to timestep"""
            noisy_logits = get_logits(noisy_embeddings, timestep)
            expected = model.embed_output_logits(noisy_logits)
            return (noisy_embeddings - expected) / math.exp(timestep)

        noise_levels = noise_schedule.get_discrete_stdevs(test_config.eval_num_steps)

        noisy_embeddings = torch.randn_like(clean_embeddings) * initial_stdev
        timesteps = [math.log(stdev) for stdev in noise_levels]
        noisy_embeddings = sample(
            noisy_embeddings, timesteps, ODE, sampler=test_config.sampler
        )
        t_final = timesteps[-1]
        predictions = get_logits(noisy_embeddings, t_final).argmax(-1)
        assert predictions.shape == (batch_size, (sequence_length - 1))

        correct = predictions == token_ids[:, 1:]
        prediction_targets = masked_positions[:, 1:] & masked_positions[:, :-1]
        correct = correct & prediction_targets

        num_correct += correct.sum().item()
        num_examples += prediction_targets.sum().item()

        progress.set_postfix({"acc": num_correct / num_examples})
        if wandb.run is not None:
            wandb.log(
                {
                    "eval/batch_idx": batch_idx,
                    "eval/running_acc": num_correct / num_examples,
                }
            )

    if wandb.run is not None and num_examples > 0:
        wandb.summary["eval/acc"] = num_correct / num_examples


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="deploy_ar_dit_simple.py")

    parser.add_argument("action")
    parser.add_argument("-i", "--load-ckpt")
    parser.add_argument("-o", "--save-ckpt")
    parser.add_argument("-l", "--log-path")
    parser.add_argument("-c", "--config")

    args = parser.parse_args()

    if args.action not in ["train", "evaluate"]:
        raise ValueError("Must choose train or evaluate")
    action = args.action

    default_output_path = (
        Path.cwd() / "outputs" / f"run-{datetime.now():%Y-%m-%d_%H-%M-%S}"
    )

    config_save_path = default_output_path / "config.yaml"

    checkpoint_save_path = default_output_path / "checkpoint"
    if args.save_ckpt is not None:
        checkpoint_save_path = Path(args.save_ckpt)

    checkpoint_load_path = args.load_ckpt

    metrics_save_path = default_output_path / "metrics.csv"
    if args.log_path is not None:
        metrics_save_path = Path(args.log_path)

    if action == "evaluate" and checkpoint_load_path is None:
        raise ValueError("Must provide model for evaluation")

    run_config = LlamaDiTRunConfig()
    if args.config is not None:
        run_config = load_config(args.config)

    wandb.init(
        project="dblocks-tinystories",
        job_type=action,
        config=asdict(run_config),
    )

    train_dataloader, test_dataloader = load_data(run_config.data_config)

    noise_schedule = NoiseLevelDistribution(**asdict(run_config.noise_config))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if checkpoint_load_path is None:
        # Block boundaries must follow the configured noise distribution.
        run_config.model_config.block_boundaries = [
            float(b)
            for b in noise_schedule.get_block_stdevs(run_config.model_config.num_blocks)
        ]
        model = LlamaDiTForCausalLM(run_config.model_config)
    else:
        model = LlamaDiTForCausalLM.from_pretrained(checkpoint_load_path)

    if action == "train":
        if run_config.train_config is None:
            raise ValueError("train_config must not be none when training")
        results_dict = train_model(
            model=model,
            dataloader=train_dataloader,
            device=device,
            noise_schedule=noise_schedule,
            train_config=run_config.train_config,
        )

        model.save_pretrained(checkpoint_save_path, max_shard_size="1GB")

        Path(metrics_save_path).parent.mkdir(parents=True, exist_ok=True)
        pandas.DataFrame(results_dict).to_csv(metrics_save_path)

        save_config(config_save_path, run_config)

    if action == "evaluate":
        if run_config.test_config is None:
            raise ValueError("test_config must not be none when testing")

        evaluate_model(
            model=model,
            dataloader=test_dataloader,
            device=device,
            noise_schedule=noise_schedule,
            test_config=run_config.test_config,
        )

    wandb.finish()

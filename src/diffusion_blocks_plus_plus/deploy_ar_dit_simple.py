import argparse
import itertools
import math
from datetime import datetime
from pathlib import Path

import pandas
import torch
import torch.nn.functional as F
from datasets import dataset_dict, load_dataset
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer, LlamaTokenizerFast
from transformers.modeling_outputs import CausalLMOutputWithPast

from .llama.attention_masks import (
    build_train_attention_mask,
    mask_to_pos,
)
from .llama.config import (
    LlamaDataConfig,
    LlamaDiTRunConfig,
    LlamaDiTRunModelConfig,
    LlamaTestConfig,
    LlamaTrainConfig,
    load_config,
    save_config,
)
from .llama.data import (
    predictable_dataset,
    prepare_training_batch,
)
from .llama.model import LlamaDiTConfig, LlamaDiTForCausalLM
from .llama.noise_level_manager import NoiseLevelDistribution
from .llama.vram_tracker import MemoryTracker


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


def train_block(
    model: LlamaDiTForCausalLM,
    block_idx: int,
    dataloader: DataLoader,
    device: torch.device,
    noise_schedule: NoiseLevelDistribution,
    train_config: LlamaTrainConfig,
    model_config: LlamaDiTRunModelConfig,
):
    model.set_primary_device(device)
    model.set_active_block(block_idx)

    current_block = model.blocks[block_idx]
    assert current_block is not None
    current_block.train()
    model.input_embedding.train()

    min_stdev, max_stdev = model.noise_boundaries(block_idx)

    optimizer = AdamW(model.block_parameters(block_idx), lr=train_config.learning_rate)

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
            )
        )

        outputs: CausalLMOutputWithPast = current_block(
            inputs_embeds=features,
            attention_mask=attention,
            position_ids=position_ids,
            **(
                {
                    "timestep": build_timesteps(
                        features, sigmas.log(), model_config.timestep_conditioning
                    )
                }
                if model_config.embed_timesteps
                else {}
            ),
        )

        memory_tracker.measure_mem_act()

        logits = outputs.logits
        assert isinstance(logits, torch.Tensor)
        assert logits.shape == (batch_size, sequence_length * 2 - 2, config.vocab_size)

        loss = F.cross_entropy(logits.permute(0, 2, 1), labels, reduction="none")
        loss = (loss * labels_mask).sum() / labels_mask.sum()
        loss.backward()

        memory_tracker.measure_mem_grad(block_idx)

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
    train_config: LlamaTrainConfig,
    model_config: LlamaDiTRunModelConfig,
) -> list[dict]:
    print(
        f"Training model on {len(dataloader.dataset)} samples for {train_config.num_epochs} epochs."  # type: ignore[reportArgumentType]
    )

    results = []

    for block_idx in tqdm(range(model.num_blocks - 1, -1, -1), desc="Block"):
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
                model_config=model_config,
            )
            results += [{"block_idx": block_idx, **row} for row in block_results]

    return results


def load_data(data_config: LlamaDataConfig):
    if data_config.dataset_name == "randb":
        train_dataset = predictable_dataset(
            data_config.num_examples_train,
            data_config.sequence_length,
        )
        test_dataset = predictable_dataset(
            data_config.num_examples_test,
            data_config.sequence_length,
        )
    else:
        tokenizer = AutoTokenizer.from_pretrained(data_config.tokenizer_config.name)
        assert isinstance(tokenizer, LlamaTokenizerFast)
        tokenizer.padding_side = data_config.tokenizer_config.padding_side
        tokenizer.add_bos_token = data_config.tokenizer_config.add_bos_token
        tokenizer.add_eos_token = data_config.tokenizer_config.add_eos_token

        def tokenize_function(examples):
            result = tokenizer(
                examples["text"],
                padding="max_length",
                truncation=True,
                max_length=data_config.sequence_length,
            )
            return result

        train_split = f"train[:{data_config.num_examples_train}]"
        train_dataset = load_dataset(data_config.dataset_name, split=train_split)
        train_dataset = train_dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=["text"],
        )

        test_split = f"validation[:{data_config.num_examples_test}]"
        test_dataset = load_dataset(data_config.dataset_name, split=test_split)
        test_dataset = test_dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=["text"],
        )

    train_dataset = train_dataset.with_format("torch")
    test_dataset = test_dataset.with_format("torch")
    assert isinstance(train_dataset, dataset_dict.Dataset)
    assert isinstance(test_dataset, dataset_dict.Dataset)

    train_dataloader = DataLoader(
        train_dataset,  # type: ignore The dataset is simple enough to be compatible
        batch_size=data_config.batch_size_train,
    )
    test_dataloader = DataLoader(
        test_dataset,  # type: ignore The dataset is simple enough to be compatible
        batch_size=data_config.batch_size_test,
    )

    return train_dataloader, test_dataloader


@torch.no_grad
def evaluate_model(
    model: LlamaDiTForCausalLM,
    dataloader: DataLoader,
    device: torch.device,
    noise_schedule: NoiseLevelDistribution,
    test_config: LlamaTestConfig,
    model_config: LlamaDiTRunModelConfig,
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

        attention_mask = build_train_attention_mask(masked_positions)
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
            current_block_idx = model.get_block_from_timestep(timestep)

            model.set_active_block(current_block_idx)

            current_block = model.blocks[current_block_idx]
            assert current_block is not None
            current_block.eval()
            model.input_embedding.eval()

            inputs_embeds = torch.cat([clean_embeddings, noisy_embeddings], dim=1)

            outputs: CausalLMOutputWithPast = current_block(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                position_ids=position_ids,
                **{
                    "timestep": build_timesteps(
                        inputs_embeds, timestep, model_config.timestep_conditioning
                    )
                }
                if model_config.embed_timesteps
                else {},
            )

            logits = outputs.logits
            assert logits is not None
            return logits[:, sequence_length - 1 :, :]

        @torch.no_grad
        def ODE(noisy_embeddings: torch.Tensor, timestep: float) -> torch.Tensor:
            """Get the change in hidden embedding with respect to timestep"""
            noisy_logits = get_logits(noisy_embeddings, timestep)
            expected = model.embed_output_logits(noisy_logits)
            # Our model is that the Gaussians scale at x=e^t, so the time derivative dx/dt is x=e^t.
            return noisy_embeddings - expected

        noise_levels = noise_schedule.get_discrete_stdevs(test_config.eval_num_steps)

        noisy_embeddings = torch.randn_like(clean_embeddings) * initial_stdev
        for stdev, next_stdev in tqdm(
            list(itertools.pairwise(noise_levels)),
            desc="Denoise Step",
            leave=False,
        ):
            dt = math.log(next_stdev) - math.log(stdev)
            noisy_embeddings += ODE(noisy_embeddings, math.log(stdev)) * dt
        t_final = math.log(noise_levels[-1])
        predictions = get_logits(noisy_embeddings, t_final).argmax(-1)
        assert predictions.shape == (batch_size, (sequence_length - 1))

        correct = predictions == token_ids[:, 1:]
        prediction_targets = masked_positions[:, 1:] & masked_positions[:, :-1]
        correct = correct & prediction_targets

        num_correct += correct.sum().item()
        num_examples += prediction_targets.sum().item()

        progress.set_postfix({"acc": num_correct / num_examples})


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

    train_dataloader, test_dataloader = load_data(run_config.data_config)

    noise_schedule = NoiseLevelDistribution()

    # See Appendix E.4 in the Sakana.ai paper
    # TODO: Generate this in an organized way from model_config
    config = LlamaDiTConfig(
        vocab_size=32000,
        num_hidden_layers=run_config.model_config.num_hidden_layers,
        hidden_size=run_config.model_config.hidden_size,
        num_attention_heads=12,
        intermediate_size=3072,
        num_blocks=run_config.model_config.num_blocks,
        use_timestep_embedding=run_config.model_config.embed_timesteps,
        timestep_conditioning=run_config.model_config.timestep_conditioning,
        block_boundaries=noise_schedule.get_block_stdevs(
            run_config.model_config.num_blocks
        ),
        apply_freq_fix=run_config.model_config.apply_freq_fix,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if checkpoint_load_path is None:
        model = LlamaDiTForCausalLM(config)
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
            model_config=run_config.model_config,
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
            model_config=run_config.model_config,
        )

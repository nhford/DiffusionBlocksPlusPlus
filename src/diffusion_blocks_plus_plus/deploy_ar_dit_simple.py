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

from diffusion_blocks_plus_plus.llama.data import DiffusionDataGenerator
from diffusion_blocks_plus_plus.llama.vram_tracker import MemoryTracker

from .llama.model import LlamaDiTConfig, LlamaDiTForCausalLM
from .plot_data import generate_plot

SEQUENCE_LENGTH = 128  # TODO: 256 is prod
BATCH_SIZE = 2  # TODO: Should be 256 as in original paper
LEARNING_RATE = 0.001
EMBED_TIMESTEPS = False

DEBUG_LEARN_NOISE = True
DEBUG_NOISE_MIN = 0.5
DEBUG_NOISE_MAX = 1
DEBUG_SMALL_DATASET = True
DEBUG_FORCE_COND = False
DEBUG_LOG_VRAM = True


def train_block(model: LlamaDiTForCausalLM, block_idx: int, dataloader: DataLoader):
    for i in range(model.num_blocks):
        model.unload_block(i)
    model.load_block(block_idx)

    min_stdev, max_stdev = model.noise_boundaries(block_idx)

    data_preprocessor = DiffusionDataGenerator()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to_device(device)

    total_params = sum(p.numel() for p in model.loaded_parameters())

    print(f"Training block {block_idx} (sigma in [{min_stdev}, {max_stdev}]).")
    print(f"Training {total_params/1e9:.2f}B parameters.")

    optimizer = AdamW(model.loaded_parameters(), lr=LEARNING_RATE)

    memory_tracker = MemoryTracker(model, optimizer, device)

    results = []

    progress = tqdm(dataloader)
    for batch_idx, batch in enumerate(progress):
        memory_tracker.reset()
        memory_tracker.measure_mem_weights()
        memory_tracker.begin_forward()

        batch_size = len(batch["input_ids"])
        token_ids: torch.Tensor = batch["input_ids"].to(device)
        assert isinstance(token_ids, torch.Tensor)
        assert token_ids.shape == (batch_size, SEQUENCE_LENGTH + 1)
        if DEBUG_LEARN_NOISE:
            token_ids = torch.randint_like(token_ids, 2)

        masked_positions: torch.Tensor = batch["attention_mask"].to(device)
        assert isinstance(masked_positions, torch.Tensor)
        assert masked_positions.shape == (batch_size, SEQUENCE_LENGTH + 1)
        if DEBUG_LEARN_NOISE:
            masked_positions = torch.ones_like(masked_positions)

        features, position_ids, sigmas, attention, labels, labels_mask = (
            data_preprocessor.prepare_training_batch(
                input_ids=token_ids,
                input_embeddings=model.input_embedding(token_ids),
                token_mask=masked_positions,
                min_stdev=min_stdev,
                max_stdev=max_stdev,
            )
        )

        current_block = model.blocks_loaded[block_idx]
        assert current_block is not None

        outputs: CausalLMOutputWithPast = current_block(
            inputs_embeds=features,
            attention_mask=attention,
            position_ids=position_ids,
            **(dict(timestep=sigmas.log()) if EMBED_TIMESTEPS else {}),
        )

        memory_tracker.measure_mem_act()

        logits = outputs.logits
        assert isinstance(logits, torch.Tensor)
        assert logits.shape == (batch_size, SEQUENCE_LENGTH * 2, config.vocab_size)

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
        progress.set_postfix({"loss": loss.item(), **memory_tracker.tqdm_dict()})
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

    for i in range(model.num_blocks - 1, -1, -1):
        results = train_block(model, i, dataloader)

        name = datetime.now().isoformat() + "_" + str(i)
        os.makedirs("./data", exist_ok=True)
        results = pandas.DataFrame(results)
        results.to_csv(f"./data/{name}.csv")

        generate_plot(name)

        model.save_checkpoint(f"./checkpoints/{name}")

from dataclasses import asdict, dataclass, field
from os import PathLike

import dacite
import yaml


@dataclass
class LlamaTokenizerConfig:
    """Configuration for Tokenizer"""

    name: str = "NousResearch/Llama-2-7b-chat-hf"
    padding_side: str = "right"
    add_eos_token: bool = True
    add_bos_token: bool = True


@dataclass
class LlamaDataConfig:
    """Configuration for Llama Datasets

    Attributes:
        dataset_name: The Huggingface ID of the dataset being loaded.
            Also accepts "randb" for random binary strings.
    """

    dataset_name: str = "roneneldan/TinyStories"
    tokenizer_config: LlamaTokenizerConfig = field(default_factory=LlamaTokenizerConfig)
    sequence_length: int = 256
    num_examples_train: int = 10000
    batch_size_train: int = 256
    num_examples_test: int = 1000
    batch_size_test: int = 256


@dataclass
class LlamaTrainConfig:
    """Configuration for Training Run"""

    learning_rate: float = 0.001
    num_epochs: int = 1


@dataclass
class LlamaTestConfig:
    """Configuration for Test Run"""

    eval_num_steps: int = 10
    sampler: str = "euler"


@dataclass
class LlamaDiTRunModelConfig:
    """The subset of LlamaDiT parameters that can be configured from yaml"""

    hidden_size: int = 768
    num_hidden_layers: int = 12
    num_blocks: int = 3
    embed_timesteps: bool = True
    timestep_conditioning: str = "uniform"
    apply_freq_fix: bool = True
    block_diffusion_size: int = 1


@dataclass
class LlamaDiTRunConfig:
    """Configuration for Llama"""

    data_config: LlamaDataConfig = field(default_factory=LlamaDataConfig)
    model_config: LlamaDiTRunModelConfig = field(default_factory=LlamaDiTRunModelConfig)
    train_config: LlamaTrainConfig | None = field(default_factory=LlamaTrainConfig)
    test_config: LlamaTestConfig | None = field(default_factory=LlamaTestConfig)


def load_config(file: str | PathLike) -> LlamaDiTRunConfig:
    with open(file, "r") as f:
        config = dacite.from_dict(
            LlamaDiTRunConfig, yaml.safe_load(f), config=dacite.Config(strict=True)
        )
    return config


def save_config(file: str | PathLike, config: LlamaDiTRunConfig) -> None:
    with open(file, "w") as f:
        yaml.safe_dump(asdict(config), f)

import json
from dataclasses import asdict, dataclass, field
from os import PathLike

import yaml
from dacite import Config, from_dict

from .model import LlamaDiTConfig


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
        num_examples_train: Number of training examples to use. None uses the
            full train partition.
        num_examples_test: Number of test examples to use. None uses the full
            validation partition.
    """

    dataset_name: str = "roneneldan/TinyStories"
    tokenizer_config: LlamaTokenizerConfig = field(default_factory=LlamaTokenizerConfig)
    sequence_length: int = 256
    num_examples_train: int | None = None
    batch_size_train: int = 256
    num_examples_test: int | None = None
    batch_size_test: int = 256


@dataclass
class NoiseLevelConfig:
    """Configuration for the noise level distribution"""

    p_mean: float = -1.2
    p_std: float = 1.2
    gamma: float = 0.1
    stdev_min: float = 0.002
    stdev_max: float = 80.0


@dataclass
class LlamaTrainConfig:
    """Configuration for Training Run"""

    learning_rate: float = 3e-4
    warmup_steps: int = 100
    num_epochs: int = 10
    bd3_lm_block_size: int = 1


@dataclass
class LlamaTestConfig:
    """Configuration for Test Run"""

    eval_num_steps: int = 4
    sampler: str = "euler"
    bd3_lm_block_size: int = 1


@dataclass
class LlamaDiTRunConfig:
    """Configuration for Llama"""

    data_config: LlamaDataConfig = field(default_factory=LlamaDataConfig)
    model_config: LlamaDiTConfig = field(default_factory=LlamaDiTConfig)
    noise_config: NoiseLevelConfig = field(default_factory=NoiseLevelConfig)
    train_config: LlamaTrainConfig | None = field(default_factory=LlamaTrainConfig)
    test_config: LlamaTestConfig | None = field(default_factory=LlamaTestConfig)


def load_config(file: str | PathLike) -> LlamaDiTRunConfig:
    with open(file, "r") as f:
        config = from_dict(
            LlamaDiTRunConfig,
            yaml.safe_load(f),
            config=Config(
                type_hooks={LlamaDiTConfig: LlamaDiTConfig.from_dict}, strict=True
            ),
        )
    return config


def save_config(file: str | PathLike, config: LlamaDiTRunConfig) -> None:
    with open(file, "w") as f:
        data = asdict(config)
        data["model_config"] = json.loads(
            config.model_config.to_json_string(use_diff=True)
        )
        yaml.safe_dump(data, f)

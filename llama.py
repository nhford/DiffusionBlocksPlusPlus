import argparse
import datetime
import json
import os

import torch
import torchmetrics
from transformers.models.llama import LlamaForCausalLM, LlamaConfig
from transformers import AutoTokenizer
import lightning as L
from transformers.optimization import get_scheduler
import torch.nn.functional as F
from torch.utils.data import DataLoader
from datasets import load_dataset
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
from lightning.pytorch.strategies import DDPStrategy, DeepSpeedStrategy


class TextDataModule(L.LightningDataModule):
    data_name = None
    dataset_kwargs = {}

    def __init__(
        self,
        batch_size: int = 1,
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


def load_llama():
    # See Appendix E.4 in the Sakana.ai paper
    config = LlamaConfig(
        vocab_size=32000,
        num_hidden_layers=12,
        hidden_size=768,
        num_attention_heads=12,
    )

    return LlamaForCausalLM(config)


class LlamaModel(L.LightningModule):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.valid_metrics = torchmetrics.MetricCollection(
            {
            },
            prefix="val/",
        )
        self.test_metrics = self.valid_metrics.clone(prefix="test/")
        self.save_hyperparameters(args)

    def configure_model(self):
        self.model = load_llama()
        self.tokenizer = AutoTokenizer.from_pretrained(
            "NousResearch/Llama-2-7b-chat-hf"
        )
        print(self.model)
        if self.args.gradient_checkpointing:
            self.model.gradient_checkpointing_enable()

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.args.lr,
            weight_decay=self.args.weight_decay,
        )
        scheduler = get_scheduler(
            name=self.args.scheduler_type,
            optimizer=optimizer,
            num_warmup_steps=self.args.num_warmup_steps,
            num_training_steps=self.trainer.estimated_stepping_batches,
            scheduler_specific_kwargs=self.args.scheduler_specific_kwargs,
        )
        return [optimizer], [{"scheduler": scheduler, "interval": "step"}]

    def forward(self, **kwargs):
        return self.model(**kwargs).logits

    def shared_step(self, batch, step="train", return_metrics=False, **kwargs):
        input_ids = batch["input_ids"]
        labels = batch["labels"]
        attention_mask = batch["attention_mask"]
        logits = self(input_ids=input_ids, attention_mask=attention_mask, **kwargs)
        if return_metrics:
            if step == "val":
                return self.valid_metrics(
                    logits.view(-1, self.num_labels), labels.view(-1)
                )
            elif step == "test":
                return self.test_metrics(
                    logits.view(-1, self.num_labels), labels.view(-1)
                )
            else:
                raise NotImplementedError(f"Step {step} is not supported")

        print(f"{logits.shape=}")
        print(f"{labels.shape=}")
        loss: torch.Tensor = F.cross_entropy(
            input=logits.view(-1, 32000),
            target=labels.view(-1),
        ) * attention_mask.view(-1)
        loss = loss.sum() / (attention_mask.sum())
        loss_dict = {f"{step}/loss": loss}
        return loss, loss_dict

    def get_model_kwargs(self, batch):
        return {}

    def training_step(self, batch, batch_idx):
        tokenizer_outputs = self.tokenizer(
            batch["text"],
            padding="max_length",
            truncation=True,
            max_length=256 + 1,
            return_tensors="pt",
        )

        batch["input_ids"] = tokenizer_outputs.input_ids[:, :-1].to(self.model.device)
        batch["labels"] = tokenizer_outputs.input_ids[:, 1:].to(self.model.device)
        batch["attention_mask"] = tokenizer_outputs.attention_mask[:, :-1].to(
            self.model.device
        )

        batch_size = len(batch["text"])
        model_kwargs = self.get_model_kwargs(batch)
        loss, loss_dict = self.shared_step(batch, step="train", **model_kwargs)
        self.log_dict(loss_dict, batch_size=batch_size, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        batch_size = batch["input_ids"].shape[0]
        model_kwargs = self.get_model_kwargs(batch)
        res = self.shared_step(batch, step="val", return_metrics=True, **model_kwargs)
        self.log_dict(res, batch_size=batch_size, prog_bar=True)

    def test_step(self, batch, batch_idx):
        batch_size = batch["input_ids"].shape[0]
        model_kwargs = self.get_model_kwargs(batch)
        res = self.shared_step(batch, step="test", return_metrics=True, **model_kwargs)
        self.log_dict(res, batch_size=batch_size, prog_bar=True)


def load_model(args):
    if args.model_type == "llama":
        return LlamaModel(args)
    else:
        raise ValueError(f"Invalid model type: {args.model_type}")


def smoke_test():
    print("Smoke test for transformers Llama implementation\n\n")

    # See Appendix E.4 in the Sakana.ai paper
    config = LlamaConfig(
        vocab_size=32000,
        num_hidden_layers=12,
        hidden_size=768,
        num_attention_heads=12,
    )

    model = LlamaForCausalLM(config)
    print(f"{model.model.embed_tokens.weight.shape=}")

    total_params = sum(p.numel() for p in model.parameters())
    print(f"{total_params=}")

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"{trainable_params=}")

    tokenizer = AutoTokenizer.from_pretrained("NousResearch/Llama-2-7b-chat-hf")

    test_prompt = "Hello, World"

    prompt_tensor = tokenizer(
        test_prompt,
        padding="max_length",
        truncation=True,
        max_length=256,
        return_tensors="pt",
    )
    print(f"{prompt_tensor.input_ids.shape=}")

    outputs = model(
        input_ids=prompt_tensor.input_ids,
        attention_mask=prompt_tensor.attention_mask,
    )
    print(f"{outputs.logits.shape=}")


def main(args):
    L.seed_everything(args.seed)

    data = load_data(args)
    model = load_model(args)
    if args.ckpt_path is not None:
        nowname = os.path.basename(os.path.dirname(args.ckpt_path))
    else:
        now = datetime.datetime.now(
            tz=datetime.timezone(datetime.timedelta(hours=9), name="EST")
        ).strftime("%Y-%m-%dT%H-%M-%S")
        nowname = now + f"-{args.model_type}" + args.postfix
        if nowname.startswith("_"):
            nowname = nowname[1:]
    print("Experiment Name:", nowname)
    logdir = os.path.join("logs", nowname)
    logger = WandbLogger(
        project=f"dblocks-llama-{args.data_name}",
        name=nowname,
        version=nowname,
        offline=args.debug,
        save_dir=logdir,
        # group=f"{args.data_name}",
    )
    trainer = L.Trainer(
        max_epochs=(
            args.num_epochs
            if args.model_type != "dblock"
            else args.num_epochs * args.num_blocks
        ),  # to align total number of iterations across the entire network because one step corresponds to one block
        check_val_every_n_epoch=args.save_every_n_epochs,
        callbacks=[
            ModelCheckpoint(
                dirpath=logdir,
                monitor="val/acc" if data.val_key is not None else None,
                mode="max",
                save_top_k=args.save_top_k,
                save_on_train_epoch_end=True,
                every_n_epochs=(
                    args.save_every_n_epochs if data.val_key is None else None
                ),
                save_last=True,
            ),
            LearningRateMonitor(logging_interval="step"),
        ],
        accumulate_grad_batches=args.accumulate_grad_batches,
        gradient_clip_val=1.0,
        strategy=(
            DDPStrategy(find_unused_parameters=args.model_type == "dblock")
            if args.devices > 1
            else "auto"
        ),
        devices=args.devices,
        logger=logger,
        num_sanity_val_steps=0,
        precision="bf16-mixed",
    )
    if args.stage == "train":
        trainer.fit(model, data, ckpt_path=args.ckpt_path)
        if data.test_key is not None:
            trainer.test(model, data.test_dataloader(), ckpt_path="best")
    else:
        assert args.ckpt_path is not None
        trainer.test(model, data, ckpt_path=args.ckpt_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", type=str, default="train", choices=["train", "test"])
    parser.add_argument("data_name", type=str, default="cifar100")
    parser.add_argument("--model_type", type=str, default="llama", choices=["llama"])
    parser.add_argument("--devices", type=int, default=1)
    parser.add_argument("--num_epochs", type=int, default=500)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--add_rand_aug", action="store_true")
    parser.add_argument("--eval_batch_size", type=int, default=None)
    parser.add_argument("--save_every_n_epochs", type=int, default=5)
    parser.add_argument("--accumulate_grad_batches", type=int, default=1)
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--scheduler_type", type=str, default="constant_with_warmup")
    parser.add_argument(
        "--scheduler_specific_kwargs",
        type=json.loads,
        default=None,
        help="specific kwargs for the scheduler",
    )
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--optimizer", type=str, default="adamw")
    parser.add_argument("--num_warmup_steps", type=int, default=0)
    parser.add_argument("--deepspeed", action="store_true", help="use deepspeed")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save_top_k", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--ckpt_path", type=str, default=None)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument(
        "--postfix", type=str, default="", help="postfix for the experiment name"
    )
    # dblock
    parser.add_argument("--num_blocks", type=int, default=3)
    parser.add_argument("--gamma", type=float, default=0.05)
    parser.add_argument("--num_inference_steps", type=int, default=None)
    parser.add_argument("--cfg_scale", type=float, default=0.0)
    parser.add_argument("--class_dropout_prob", type=float, default=0.0)
    args = parser.parse_args()
    main(args)

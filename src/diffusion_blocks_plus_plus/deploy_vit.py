import os
import argparse
import datetime
import json
import torch
import lightning as L
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
from lightning.pytorch.strategies import DDPStrategy, DeepSpeedStrategy

from .vit.data import load_data
from .vit.model import load_model

torch.set_float32_matmul_precision("high")
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


class SaveMilestones(L.Callback):
    def __init__(self, directory, epochs, stop_after):
        self.directory = directory
        self.epochs = set(epochs)
        self.stop_after = stop_after

    def on_train_epoch_end(self, trainer, pl_module):
        epoch = trainer.current_epoch + 1
        if epoch in self.epochs or epoch == self.stop_after:
            trainer.save_checkpoint(os.path.join(self.directory, f"milestone-{epoch:03d}.ckpt"))
        if epoch == self.stop_after:
            trainer.should_stop = True


def main(args):
    L.seed_everything(args.seed)

    data = load_data(args)
    args.image_size = data.image_size
    args.num_labels = data.num_labels
    model = load_model(args)
    if args.ckpt_path is not None:
        nowname = os.path.basename(os.path.dirname(args.ckpt_path))
    else:
        now = datetime.datetime.now(
            tz=datetime.timezone(datetime.timedelta(hours=9), name="JST")
        ).strftime("%Y-%m-%dT%H-%M-%S")
        nowname = now + f"-{args.model_type}" + args.postfix
        if nowname.startswith("_"):
            nowname = nowname[1:]
    run_name = nowname
    if args.stage == "test":
        run_name = f"{args.data_name}-{args.model_type}"
        if args.model_type == "dblock":
            run_name += f"-{args.ode}-{model.num_inference_steps}pts"
        run_name += f"-seed{args.seed}{args.postfix}"

    print("Experiment Name:", run_name)
    logdir = os.path.join("logs", nowname)
    logger = WandbLogger(
        project=os.environ.get("WANDB_PROJECT", f"dblocks-{args.data_name}"),
        name=run_name,
        version=os.environ.get("WANDB_RUN_ID") or __import__("uuid").uuid4().hex,
        log_model=True,
        offline=args.debug,
        save_dir=logdir,
        # group=f"{args.data_name}",
    )
    max_epochs = args.num_epochs
    if args.model_type == "dblock" and not args.dblock_literal_epochs:
        max_epochs *= args.num_blocks
    trainer = L.Trainer(
        max_epochs=max_epochs,
        check_val_every_n_epoch=args.save_every_n_epochs,
        callbacks=[
            ModelCheckpoint(
                dirpath=logdir,
                monitor="val/acc" if data.val_key is not None else None,
                mode="max",
                save_top_k=args.save_top_k,
                save_on_train_epoch_end=data.val_key is None,
                every_n_epochs=args.save_every_n_epochs
                if data.val_key is None
                else None,
                save_last=True,
            ),
            LearningRateMonitor(logging_interval="step"),
            SaveMilestones(logdir, args.checkpoint_epochs, args.stop_after_epoch),
        ],
        accumulate_grad_batches=args.accumulate_grad_batches,
        gradient_clip_val=1.0,
        strategy=DDPStrategy(find_unused_parameters=args.model_type == "dblock")
        if args.devices > 1
        else "auto",
        devices=args.devices,
        logger=logger,
        num_sanity_val_steps=0,
        precision=args.precision,
    )
    if args.stage == "train":
        trainer.fit(model, data, ckpt_path=args.ckpt_path)
        if (
            data.test_key is not None
            and not args.skip_test_after_fit
            and args.stop_after_epoch is None
        ):
            trainer.test(
                model, data.test_dataloader(),
                ckpt_path="best" if data.val_key is not None else "last",
            )
    else:
        assert args.ckpt_path is not None
        model.test_kwargs = (
            {"ode": args.ode} if args.model_type == "dblock" else {}
        )
        trainer.test(model, data, ckpt_path=args.ckpt_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", type=str, default="train", choices=["train", "test"])
    parser.add_argument("data_name", type=str, default="cifar100")
    parser.add_argument(
        "--model_type", type=str, default="vit", choices=["vit", "dblock"]
    )
    parser.add_argument("--devices", type=int, default=1)
    parser.add_argument("--num_epochs", type=int, default=500)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--validation_fraction", type=float, default=0.0)
    parser.add_argument("--precision", type=str, default="32-true",
                        choices=["32-true", "bf16-mixed", "16-mixed"])
    parser.add_argument("--add_rand_aug", action="store_true")
    parser.add_argument("--no_padded_crop", action="store_true", help="Disable the padded CIFAR-100 training crop")
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
    warmup = parser.add_mutually_exclusive_group()
    warmup.add_argument("--num_warmup_steps", type=int, default=0)
    warmup.add_argument("--num_warmup_epochs", type=int)
    parser.add_argument("--checkpoint_epochs", type=int, nargs="+", default=[])
    parser.add_argument("--stop_after_epoch", type=int, help="Save and pause while keeping the full LR schedule")
    parser.add_argument("--deepspeed", action="store_true", help="use deepspeed")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save_top_k", type=int, default=1)
    parser.add_argument("--skip_test_after_fit", action="store_true",
                        help="Finish after training/checkpointing without running test evaluation")
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--ckpt_path", type=str, default=None)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument(
        "--postfix", type=str, default="", help="postfix for the experiment name"
    )
    # dblock
    parser.add_argument("--num_blocks", type=int, default=3)
    parser.add_argument("--dblock_literal_epochs", action="store_true", help="Count actual loader passes for DiffusionBlocks")
    parser.add_argument("--dblock_paper_grid", action="store_true", help="Route grid boundaries to the lower-noise block")
    parser.add_argument(
        "--invert_timestep_frequencies",
        action="store_true",
        help="Use the Llama-style ascending frequency bank for DiffusionBlocks",
    )
    parser.add_argument("--dblock_plain_ce", action="store_true", help="Use unweighted cross-entropy for DiffusionBlocks")
    parser.add_argument("--gamma", type=float, default=0.05)
    parser.add_argument("--num_inference_steps", type=int, default=None)
    parser.add_argument("--cfg_scale", type=float, default=0.0)
    parser.add_argument("--class_dropout_prob", type=float, default=0.0)
    parser.add_argument("--ode", type=str, default="euler",
                        choices=["euler", "heun", "midpoint"])
    args = parser.parse_args()
    if args.model_type == "dblock":
        if args.num_blocks < 1 or 12 % args.num_blocks:
            parser.error("num_blocks must be a positive divisor of the 12 ViT layers")
        if args.dblock_paper_grid and (args.num_inference_steps or 0) < args.num_blocks + 1:
            parser.error("dblock_paper_grid requires at least num_blocks + 1 explicit grid points")
    if any(not 1 <= epoch <= args.num_epochs for epoch in args.checkpoint_epochs):
        parser.error("checkpoint epochs must be between 1 and num_epochs")
    if args.stop_after_epoch is not None and not 1 <= args.stop_after_epoch <= args.num_epochs:
        parser.error("stop_after_epoch must be between 1 and num_epochs")
    if args.num_warmup_epochs is not None and not 0 <= args.num_warmup_epochs <= args.num_epochs:
        parser.error("num_warmup_epochs must be between 0 and num_epochs")
    main(args)

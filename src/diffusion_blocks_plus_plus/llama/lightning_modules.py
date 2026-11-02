import lightning as L
import torch
import torch.nn.functional as F
import torchmetrics
from transformers import AutoTokenizer, LlamaConfig, LlamaForCausalLM
from transformers.optimization import get_scheduler

class LlamaLightningModule(L.LightningModule):
    max_seq_length = 256
    # See Appendix E.4 in the Sakana.ai paper
    llama_config = LlamaConfig(
        vocab_size=32000,
        num_hidden_layers=12,
        hidden_size=768,
        num_attention_heads=12,
        intermediate_size=3072, # TODO: Experiment to find the correct value here, since it is not stated in the paper
    )


    def __init__(self, args):
        super().__init__()
        self.args = args
        self.valid_metrics = torchmetrics.MetricCollection(
            {},
            prefix="val/",
        )
        self.test_metrics = self.valid_metrics.clone(prefix="test/")
        self.save_hyperparameters(args)

    def configure_model(self):
        self.model = LlamaForCausalLM(self.llama_config)
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
        tokenizer_outputs = self.tokenizer(
            batch["text"],
            padding="max_length",
            truncation=True,
            max_length=self.max_seq_length + 1,
            return_tensors="pt",
        )

        input_ids = tokenizer_outputs.input_ids[:, :-1].to(self.model.device)
        labels = tokenizer_outputs.input_ids[:, 1:].to(self.model.device)
        attention_mask = tokenizer_outputs.attention_mask[:, :-1].to(self.model.device)
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

        loss: torch.Tensor = F.cross_entropy(
            input=logits.view(-1, self.llama_config.vocab_size),
            target=labels.view(-1),
        ) * attention_mask.view(-1)
        loss = loss.sum() / (attention_mask.sum())
        loss_dict = {f"{step}/loss": loss}
        return loss, loss_dict

    def get_model_kwargs(self, batch):
        return {}

    def training_step(self, batch, batch_idx):
        batch_size = len(batch["text"])
        model_kwargs = self.get_model_kwargs(batch)
        loss, loss_dict = self.shared_step(batch, step="train", **model_kwargs)
        self.log_dict(loss_dict, batch_size=batch_size, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        batch_size = len(batch["text"])
        model_kwargs = self.get_model_kwargs(batch)
        res = self.shared_step(batch, step="val", return_metrics=True, **model_kwargs)
        self.log_dict(res, batch_size=batch_size, prog_bar=True)

    def test_step(self, batch, batch_idx):
        batch_size = len(batch["text"])
        model_kwargs = self.get_model_kwargs(batch)
        res = self.shared_step(batch, step="test", return_metrics=True, **model_kwargs)
        self.log_dict(res, batch_size=batch_size, prog_bar=True)


class LlamaDBlockLightningModule(LlamaLightningModule):
    # TODO: finish implementing LlamaDBlockModel
    max_seq_length = 256
    # See Appendix E.4 in the Sakana.ai paper
    llama_config = LlamaConfig(
        vocab_size=32000,
        num_hidden_layers=12,
        hidden_size=768,
        num_attention_heads=12,
        intermediate_size=3072, # TODO: Experiment to find the correct value here, since it is not stated in the paper
    )


    def build_attention_mask(self, token_mask: torch.Tensor):
        """
        Build the attention mask for auto-regressive diffusion-blocks training.

        Since we have a non-trivial masking relation, we need to build a custom attention mask (see https://github.com/huggingface/transformers/pull/27539)
        As noted in the PR, this tensor contains 0 on the lower triangular in the top half, and copies it down for the second half (noisy tokens can attend to previous clean tokens), and a diagonal in the bottom left (noisy tokens can attend to themselves).
        Note that according to https://github.com/huggingface/transformers/pull/27539#issuecomment-2531760367, negative infinity and 0 should be used.
        For example, if we use i to represent negative infinity, if our max_seq_length is 4, then this matrix will be returned:

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

        Based on the PR, it seems like the shape for the attention mask is supposed to be (BxHxSxS), or (batch, head, query, key).

        Returns:
            A tensor of shape (1, 1, seq_length * 2, seq_length * 2) with the attention mask.

        TODO: Since sequences are left-padded, we need to also mask out the padding tokens
        """
        result = torch.full(
            (1, 1, self.max_seq_length * 2, self.max_seq_length * 2), float("-inf")
        )
        # Set lower triangular entries to 0
        result.triu_()

        # Clear left of the triangle
        for row in range(self.max_seq_length, self.max_seq_length * 2):
            result[:, :, row, row + 1 : row + self.max_seq_length] = float("-inf")

    def generate_noise_assignments(self, batch_size: int):
        """
        Generate a noise level for each batch

        For context, the way that DiffusionBlocks models are trained is that each batch trains a single block.
        Within that block, the noise levels for each instance should be uniformly distributed (w.r.t normal measure yada yada).
        However, v
        TODO: Finish Documentation

        """
        # TODO Implement correctly
        return torch.full((batch_size,), 0.1)


    def shared_step(self, batch, step="train", return_metrics=False, **kwargs):
        batch_size = len(batch)

        tokenizer_outputs = self.tokenizer(
            batch["text"],
            padding="max_length",
            truncation=True,
            max_length=self.max_seq_length + 1,
            return_tensors="pt",
        )
        input_ids = tokenizer_outputs.input_ids[:, :-1].to(self.model.device)
        labels = tokenizer_outputs.input_ids[:, 1:].to(self.model.device)
        token_mask = tokenizer_outputs.attention_mask[:, :-1].to(self.model.device)

        clean_embeds = self.model.model.embed_tokens(input_ids)
        noisy_embeds = self.model.model.embed_tokens(labels)

        # TODO add noise to noisy_embeds

        assert clean_embeds.shape == (
            batch_size,
            self.max_seq_length,
            self.llama_config.hidden_size,
        )
        assert noisy_embeds.shape == (
            batch_size,
            self.max_seq_length,
            self.llama_config.hidden_size,
        )

        inputs_embeds = torch.cat([clean_embeds, noisy_embeds], dim=1)

        # TODO add noise level embeddings into the tokens
        # In the Sakana.ai paper, they state that they "augmented with time conditioning as in DiT", citing
        # https://arxiv.org/abs/2212.09748
        # NOTE For now, we can just raw dog the rope embeddings

        attention_mask: torch.Tensor = self.build_attention_mask()
        attention_mask = attention_mask.expand((batch_size, -1, -1, -1))

        # Predict denoised tokens
        logits = self.model(
            inputs_embeds=inputs_embeds, attention_mask=attention_mask, **kwargs
        ).logits
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

        loss: torch.Tensor = F.cross_entropy(
            input=logits.view(-1, self.llama_config.vocab_size),
            target=labels.view(-1),
        ) * token_mask.view(-1)
        loss = loss.sum() / (token_mask.sum())
        loss_dict = {f"{step}/loss": loss}
        return loss, loss_dict


def load_model(args):
    if args.model_type == "llama":
        return LlamaLightningModule(args)
    else:
        raise ValueError(f"Invalid model type: {args.model_type}")


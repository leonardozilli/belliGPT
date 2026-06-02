import logging
import math
import os
from pathlib import Path

import numpy as np
import torch

import wandb
from common.config import GBTConfig, TrainConfig
from common.model import GBT

logger = logging.getLogger(__name__)


class DataLoader:
    def __init__(
        self, B: int, T: int, device: str, data_dir: str, split: str = "train"
    ):
        self.B = B
        self.T = T
        self.device = device
        self.current_position = 0

        file_path = os.path.join(data_dir, f"{split}.bin")
        self.tokens = np.memmap(file_path, dtype=np.uint16, mode="r")
        logger.info(f"loaded {len(self.tokens)} {split} tokens from {data_dir}")

        self.reset()

    def reset(self):
        self.current_position = 0

    def next_batch(self):
        B, T = self.B, self.T

        if self.current_position + B * T + 1 > len(self.tokens):
            self.reset()

        buf = (
            torch.from_numpy(
                self.tokens[
                    self.current_position : self.current_position + B * T + 1
                ].copy()
            )
            .to(self.device)
            .long()
        )
        x = buf[:-1].view(B, T)
        y = buf[1:].view(B, T)

        self.current_position += B * T

        return x, y


def get_lr(step: int, config: TrainConfig):
    lr_max = config.learning_rate
    lr_min = 0.1 * lr_max  # 10% of max

    if step < config.warmup_steps:
        # linear warmupt 0 -> lr_max
        return lr_max * (step + 1) / config.warmup_steps
    else:
        # cosine decay lr_max -> lr_min
        progress = (step - config.warmup_steps) / max(
            1, config.max_steps - config.warmup_steps
        )
        return lr_min + 0.5 * (lr_max - lr_min) * (1 + math.cos(math.pi * progress))


def sample(
    model: GBT,
    tokenizer,
    config: GBTConfig,
    prompt: str = "",
    max_new_tokens: int = 100,
):
    model.eval()
    try:
        bos_token = tokenizer.special_tokens["BOS"]
    except KeyError:
        bos_token = prompt
    input_ids = tokenizer.encode(bos_token)
    if hasattr(input_ids, "ids"):
        input_ids = input_ids.ids

    context = torch.tensor([input_ids], dtype=torch.long, device=config.device)
    pred = model.generate(context, max_new_tokens=max_new_tokens)
    generated_text = tokenizer.decode(pred[0].tolist(), skip_special_tokens=False)
    logger.info(generated_text)

    return generated_text


def eval(model: GBT, val_loader: DataLoader, step: int, config: GBTConfig):
    model.eval()
    val_loader.reset()
    with torch.no_grad():
        val_losses = []
        for _ in range(10):
            try:
                x, y = val_loader.next_batch()
            except StopIteration:
                break
            x, y = x.to(config.device), y.to(config.device)
            with torch.autocast(device_type=config.device, dtype=torch.bfloat16):
                _, loss = model(x, y)
            val_losses.append(loss.item())

        val_loss = np.mean(val_losses)
        perplexity = math.exp(val_loss)
        logger.info(f"step {step}: val loss {val_loss}, perplexity {perplexity:.2f}")

        return val_loss, perplexity


def save_model(model, optimizer, val_loss, model_config, step, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": val_loss,
            "config": model_config.__dict__,
        },
        output_dir / f"checkpoint_{step}.pt",
    )


def init_wandb(train_cfg, model_cfg, model, dataset_name, tokenizer):
    run = wandb.init(
        project="belliGPT",
        job_type="train",
        notes=train_cfg.notes,
        config={
            "architecture": "GBT",
            "dataset": dataset_name,
            "learning_rate": train_cfg.learning_rate,
            "batch_size": train_cfg.batch_size,
            "block_size": model_cfg.block_size,
            "n_layer": model_cfg.n_layer,
            "n_head": model_cfg.n_head,
            "n_embd": model_cfg.n_embd,
            "dropout": model_cfg.dropout,
            "weight_decay": train_cfg.weight_decay,
            "max_steps": train_cfg.max_steps,
            "warmup_steps": train_cfg.warmup_steps,
            "M_params": model.get_num_params() / 1e6,
            "vocab_size": model_cfg.vocab_size,
            "tokenizer_type": tokenizer.kind,
        },
        dir=train_cfg.output_dir,
    )
    wandb.watch(model, log="all", log_freq=500)
    return run


def setup_logging(log_file: str):
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(),
        ],
    )

    return logging.getLogger(__name__)

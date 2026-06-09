from pathlib import Path

from finetune.config import FinetuneConfig


def load_datasets(cfg: FinetuneConfig, tokenizer):
    from datasets import load_dataset

    train_path = cfg.train_path()
    if not Path(train_path).exists():
        raise FileNotFoundError(f"train file not found: {train_path}")
    train = load_dataset("json", data_files=train_path, split="train")

    eval_ds = None
    eval_path = cfg.eval_path()
    if eval_path and Path(eval_path).exists():
        eval_ds = load_dataset("json", data_files=eval_path, split="train")

    train = _ensure_text_column(train, cfg, tokenizer)
    if eval_ds is not None:
        eval_ds = _ensure_text_column(eval_ds, cfg, tokenizer)

    return train, eval_ds


def _append_eos(dataset, tokenizer):
    eos = tokenizer.eos_token

    def add(example):
        text = example["text"].rstrip()
        if not text.endswith(eos):
            text = text + eos
        return {"text": text}

    return dataset.map(add)


def _ensure_text_column(dataset, cfg: FinetuneConfig, tokenizer):
    if "text" in dataset.column_names:
        return _append_eos(dataset, tokenizer)
    if "messages" in dataset.column_names:
        if not cfg.use_chat_template:
            raise ValueError(
                "dataset has a `messages` column but use_chat_template is False"
            )

        def fmt(examples):
            return {
                "text": [
                    tokenizer.apply_chat_template(
                        convo, tokenize=False, add_generation_prompt=False
                    )
                    for convo in examples["messages"]
                ]
            }

        return dataset.map(fmt, batched=True)
    raise ValueError(
        f"dataset has neither `text` nor `messages` column: {dataset.column_names}"
    )


def length_stats(dataset, tokenizer) -> dict[str, int]:
    import numpy as np

    lens = dataset.map(lambda x: {"len": len(tokenizer.encode(x["text"]))})["len"]
    return {
        "max": int(max(lens)),
        "median": int(np.median(lens)),
        "mean": int(np.mean(lens)),
    }

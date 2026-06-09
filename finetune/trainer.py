import os

from finetune import data as _data
from finetune import modeling as _modeling
from finetune.config import FinetuneConfig


def build_trainer(model, tokenizer, train_dataset, eval_dataset, cfg: FinetuneConfig):
    # isort: off
    from unsloth import UnslothTrainer, UnslothTrainingArguments, is_bfloat16_supported
    from transformers import EarlyStoppingCallback
    # isort: on

    if eval_dataset is not None:
        eval_args = dict(
            eval_strategy="steps",
            eval_steps=cfg.eval_steps,
            per_device_eval_batch_size=(cfg.per_device_train_batch_size or 8) * 2,
            save_strategy="steps",
            save_steps=cfg.eval_steps,
            save_total_limit=cfg.save_total_limit,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
        )
        callbacks = [
            EarlyStoppingCallback(early_stopping_patience=cfg.early_stopping_patience)
        ]
    else:
        eval_args = dict(save_steps=50)
        callbacks = []

    return UnslothTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        callbacks=callbacks,
        dataset_text_field="text",
        max_seq_length=cfg.max_seq_length,
        packing=cfg.packing,
        args=UnslothTrainingArguments(
            per_device_train_batch_size=cfg.per_device_train_batch_size,
            gradient_accumulation_steps=cfg.gradient_accumulation_steps,
            warmup_steps=cfg.warmup_steps,
            num_train_epochs=cfg.num_train_epochs,
            learning_rate=cfg.learning_rate,
            embedding_learning_rate=cfg.embedding_learning_rate,
            seed=cfg.seed,
            fp16=not is_bfloat16_supported(),
            bf16=is_bfloat16_supported(),
            logging_steps=cfg.logging_steps,
            optim=cfg.optim,
            weight_decay=cfg.weight_decay,
            output_dir=cfg.output_dir,
            **eval_args,
        ),
    )


def run(cfg: FinetuneConfig) -> str:
    import unsloth

    cfg = cfg.resolve()
    print(f"[finetune] {cfg.summary()}")

    model, tokenizer = _modeling.load_model_and_tokenizer(cfg)
    model = _modeling.attach_lora(model, cfg)

    train_ds, eval_ds = _data.load_datasets(cfg, tokenizer)
    stats = _data.length_stats(train_ds, tokenizer)
    print(
        f"[finetune] train={len(train_ds)} eval="
        f"{len(eval_ds) if eval_ds else 'none'} | token lengths {stats} "
        f"(max_seq_length={cfg.max_seq_length})"
    )
    if stats["max"] > cfg.max_seq_length:
        print(
            f"[finetune] WARNING: longest example ({stats['max']} tokens) exceeds "
            f"max_seq_length ({cfg.max_seq_length}) — it will be truncated."
        )

    trainer = build_trainer(model, tokenizer, train_ds, eval_ds, cfg)

    os.environ["UNSLOTH_OFFLOAD_GRADIENTS"] = "0"
    unsloth.USE_MODERN_PEFT = True
    trainer.train()

    adapter_dir = cfg.adapter_dir
    model.save_pretrained(adapter_dir)
    # overwrite previous added_tokens.json if it exists
    stale_added = os.path.join(adapter_dir, "added_tokens.json")
    if os.path.exists(stale_added):
        os.remove(stale_added)
    tokenizer.save_pretrained(adapter_dir)
    print(f"[finetune] saved adapter → {adapter_dir}")
    return adapter_dir

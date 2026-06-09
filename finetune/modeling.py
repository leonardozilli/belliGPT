from finetune.config import FinetuneConfig


def load_model_and_tokenizer(cfg: FinetuneConfig):
    from transformers import AddedToken
    from unsloth import FastLanguageModel, add_new_tokens

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=cfg.model_name,
        max_seq_length=cfg.max_seq_length,
        dtype=None,
        load_in_4bit=cfg.load_in_4bit,
        fix_tokenizer=False,
    )

    if cfg.add_struct_tokens:
        add_new_tokens(
            model,
            tokenizer,
            new_tokens=[
                AddedToken(t, normalized=False, special=False)
                for t in cfg.struct_tokens
            ],
        )

    if cfg.use_chat_template:
        from unsloth.chat_templates import get_chat_template

        tokenizer = get_chat_template(tokenizer, chat_template=cfg.chat_template)

    return model, tokenizer


def attach_lora(model, cfg: FinetuneConfig):
    """Wrap the model with the LoRA adapter."""
    from unsloth import FastLanguageModel

    return FastLanguageModel.get_peft_model(
        model,
        r=cfg.lora_r,
        target_modules=cfg.target_modules,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        bias="none",
        use_gradient_checkpointing=cfg.use_gradient_checkpointing,
        random_state=cfg.seed,
        use_rslora=False,
        loftq_config=None,
    )

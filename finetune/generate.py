import re

from finetune.config import FinetuneConfig

_RHYME_PREFIX = re.compile(r"(?m)^[ \t]*\[[A-G]\][ \t]*")


def clean(text: str) -> str:
    """Strip the rhyme markers."""
    text = _RHYME_PREFIX.sub("", text)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def load_adapter(
    adapter_dir: str, max_seq_length: int = 512, load_in_4bit: bool = True
):
    """Load a saved adapter."""
    import unsloth

    model, tokenizer = unsloth.FastLanguageModel.from_pretrained(
        model_name=adapter_dir,
        max_seq_length=max_seq_length,
        device_map="cuda:0",
        dtype=None,
        load_in_4bit=load_in_4bit,
        fix_tokenizer=False,
    )
    unsloth.FastLanguageModel.for_inference(model)
    return model, tokenizer


def build_prompt(title: str | None = None) -> str:
    """Build a generation prompt for a sonnet:

    - title=None -> "TITLE:" only (model must generate the title, then the sonnet);
    - title="" -> "SONNET\\n\\n" only;
    - otherwise "TITLE: {title}\\n\\nSONNET\\n\\n".
    """
    if title is None:
        return "TITLE:"
    if title == "":
        return "SONNET\n\n"
    return f"TITLE: {title}\n\nSONNET\n\n"


def generate(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 400,
    temperature: float = 0.8,
    top_p: float = 0.95,
    repetition_penalty: float = 1.0,
    stream: bool = False,
    strip_tags: bool = False,
) -> str | None:
    from transformers import TextStreamer

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    streamer = (
        TextStreamer(tokenizer, skip_prompt=True) if stream and not strip_tags else None
    )
    out = model.generate(
        **inputs,
        streamer=streamer,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=temperature,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
    )
    if streamer is not None:
        return None
    gen = tokenizer.decode(
        out[0][inputs.input_ids.shape[1] :], skip_special_tokens=True
    )
    return clean(gen) if strip_tags else gen


def adapter_dir_for(cfg: FinetuneConfig) -> str:
    return cfg.resolve().adapter_dir

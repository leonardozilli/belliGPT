import os

from finetune.generate import clean

_HERE = os.path.dirname(__file__)
DEFAULT_GGUF = os.path.join(_HERE, "gguf", "minerva7b-base-Q4_K_M.gguf")
DEFAULT_LORA = os.path.join(_HERE, "gguf", "belli-7b-lora-f16.gguf")


def load_engine(
    gguf_path: str = DEFAULT_GGUF,
    lora_path: str | None = DEFAULT_LORA,
    n_gpu_layers: int = -1,  # offload every layer to gpu
    n_ctx: int = 1024,
    seed: int | None = None,
    verbose: bool = False,
    logits_all: bool = False,
):
    """Load the GGUF base + optional LoRA."""

    from llama_cpp import Llama

    if not os.path.exists(gguf_path):
        raise FileNotFoundError(f"GGUF not found: {gguf_path}.")
    kwargs: dict = {
        "model_path": gguf_path,
        "n_gpu_layers": n_gpu_layers,
        "n_ctx": n_ctx,
        "verbose": verbose,
        "logits_all": logits_all,
    }
    if lora_path:
        if not os.path.exists(lora_path):
            raise FileNotFoundError(f"LoRA GGUF not found: {lora_path}")
        kwargs["lora_path"] = lora_path
    if seed is not None:
        kwargs["seed"] = seed
    return Llama(**kwargs)


def generate(
    llm,
    prompt: str,
    max_new_tokens: int = 400,
    temperature: float = 0.8,
    top_p: float = 0.95,
    repetition_penalty: float = 1.0,
    strip_tags: bool = False,
) -> str:
    """Sample one completion."""
    out = llm.create_completion(
        prompt,
        max_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        repeat_penalty=repetition_penalty,
    )
    gen = out["choices"][0]["text"]
    return clean(gen) if strip_tags else gen

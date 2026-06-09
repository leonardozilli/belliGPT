import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from finetune import env as _env

CONFIGS_DIR = Path(__file__).resolve().parent / "configs"
CONFIG_GROUPS = ("model", "dataset", "env")

STRUCT_TOKENS: list[str] = [
    "<TITLE>",
    "<SONNET>",
    "<STANZA>",
    "<END>",
    *[f"<RHYME_{c}>" for c in "ABCDEFG"],
]

DEFAULT_TARGET_MODULES: list[str] = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]


@dataclass
class FinetuneConfig:
    model_name: str = "sapienzanlp/Minerva-3B-base-v1.0"
    dataset_name: str = "sonnets_rhymes"
    train_file: str = "finetune_train.jsonl"
    eval_file: str | None = "finetune_eval.jsonl"

    max_seq_length: int = 512
    load_in_4bit: bool = True

    # lora config
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.0
    target_modules: list[str] = field(
        default_factory=lambda: list(DEFAULT_TARGET_MODULES)
    )
    use_gradient_checkpointing: bool | str = False

    add_struct_tokens: bool = False
    struct_tokens: list[str] = field(default_factory=lambda: list(STRUCT_TOKENS))

    use_chat_template: bool = False
    chat_template: str = "chatml"

    # optim
    learning_rate: float = 2e-4
    embedding_learning_rate: float = 2e-4  # ignored by unsloth
    num_train_epochs: int = 8
    warmup_steps: int = 14
    weight_decay: float = 0.01
    optim: str = "adamw_8bit"
    seed: int = 42

    # batch size
    per_device_train_batch_size: int | None = None
    gradient_accumulation_steps: int | None = None
    effective_batch_target: int = 64
    packing: bool = False

    # eval / checkpointing
    eval_steps: int = 10
    save_total_limit: int = 2
    early_stopping_patience: int = 3
    logging_steps: int = 1

    # storage
    env: _env.Environment | None = None
    data_dir: str | None = None
    output_dir: str | None = None

    # helpers
    @property
    def adapter_name(self) -> str:
        return f"{self.model_name.split('/')[-1]}_belli_adapter"

    @property
    def adapter_dir(self) -> str:
        assert self.output_dir is not None, "call resolve() first"
        return str(Path(self.output_dir) / self.dataset_name / self.adapter_name)

    def train_path(self) -> str:
        assert self.data_dir is not None, "call resolve() first"
        return str(Path(self.data_dir) / self.dataset_name / self.train_file)

    def eval_path(self) -> str | None:
        if not self.eval_file:
            return None
        assert self.data_dir is not None, "call resolve() first"
        return str(Path(self.data_dir) / self.dataset_name / self.eval_file)

    # build
    @classmethod
    def from_yaml(cls, path: str | Path) -> "FinetuneConfig":
        import yaml

        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "FinetuneConfig":
        known = {f.name for f in dataclasses.fields(cls)}
        unknown = set(data) - known
        if unknown:
            raise ValueError(f"unknown config keys: {sorted(unknown)}")
        return cls(**data)

    @classmethod
    def compose(
        cls,
        base: str | Path | None = None,
        group_overrides: dict[str, str] | None = None,
        field_overrides: dict[str, Any] | None = None,
    ) -> "FinetuneConfig":
        """Build a config from the Hydra group layout."""
        import yaml

        base_path = Path(base) if base else CONFIGS_DIR / "base.yaml"
        with open(base_path, "r", encoding="utf-8") as f:
            base_data = yaml.safe_load(f) or {}

        defaults = dict(base_data.pop("defaults", {}) or {})
        for group, override_choice in (group_overrides or {}).items():
            if override_choice is not None:
                defaults[group] = override_choice

        merged: dict[str, Any] = {}
        base_dir = base_path.parent
        for group in CONFIG_GROUPS:
            choice = defaults.get(group)
            if not choice or choice == "auto":
                continue
            group_path = base_dir / group / f"{choice}.yaml"
            if not group_path.exists():
                raise FileNotFoundError(
                    f"config group file not found: {group_path} "
                    f"(choices: {[p.stem for p in (base_dir / group).glob('*.yaml')]})"
                )
            with open(group_path, "r", encoding="utf-8") as f:
                merged.update(yaml.safe_load(f) or {})

        merged.update(base_data)
        merged.update(field_overrides or {})
        return cls.from_dict(merged)

    def merge(self, overrides: dict[str, Any]) -> "FinetuneConfig":
        clean = {k: v for k, v in overrides.items() if v is not None}
        return dataclasses.replace(self, **clean)

    def resolve(self) -> "FinetuneConfig":
        env = self.env or _env.detect_environment()
        data_root, output_root = _env.storage_roots(env)
        per_device = self.per_device_train_batch_size
        grad_accum = self.gradient_accumulation_steps
        if per_device is None or grad_accum is None:
            auto_pd, auto_ga = _env.auto_batch(self.effective_batch_target)
            per_device = per_device or auto_pd
            grad_accum = grad_accum or auto_ga
        return dataclasses.replace(
            self,
            env=env,
            data_dir=self.data_dir or str(data_root),
            output_dir=self.output_dir or str(output_root),
            per_device_train_batch_size=per_device,
            gradient_accumulation_steps=grad_accum,
        )

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    def summary(self) -> str:
        eff = (self.per_device_train_batch_size or 0) * (
            self.gradient_accumulation_steps or 0
        )
        return (
            f"{self.model_name} for {self.dataset_name} | env={self.env} | "
            f"batch {self.per_device_train_batch_size}x{self.gradient_accumulation_steps}"
            f" (eff {eff}) | lr {self.learning_rate} | "
            f"epochs≤{self.num_train_epochs} | eval={'on' if self.eval_file else 'off'}"
        )

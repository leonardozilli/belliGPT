import ast
import dataclasses

import click

from finetune import env as _env
from finetune.config import FinetuneConfig


def _coerce(value: str):
    try:
        return ast.literal_eval(value)
    except (ValueError, SyntaxError):
        return value


@click.command()
@click.option(
    "--config",
    "config_path",
    type=click.Path(exists=True),
    default=None,
    help="Base config file.",
)
@click.option(
    "--model",
    "model_group",
    default=None,
    help="Model group choice.",
)
@click.option(
    "--dataset",
    "dataset_group",
    default=None,
    help="Dataset group choice.",
)
@click.option(
    "--env",
    "env_group",
    default=None,
    type=click.Choice(["auto", "local", "colab", "remote"]),
    help="Env group choice (default from base: auto).",
)
@click.option("--data-dir", default=None, help="Override data root.")
@click.option("--output-dir", default=None, help="Override output root.")
@click.option("--no-eval", is_flag=True, help="Disable eval and train on all data.")
@click.option(
    "--set",
    "overrides",
    multiple=True,
    metavar="KEY=VALUE",
    help="Override any FinetuneConfig field.",
)
@click.option("--dry-run", is_flag=True)
def main(
    config_path,
    model_group,
    dataset_group,
    env_group,
    data_dir,
    output_dir,
    no_eval,
    overrides,
    dry_run,
):
    field_overrides: dict = {}
    if data_dir is not None:
        field_overrides["data_dir"] = data_dir
    if output_dir is not None:
        field_overrides["output_dir"] = output_dir
    for item in overrides:
        if "=" not in item:
            raise click.BadParameter(f"--set expects KEY=VALUE, got {item!r}")
        key, _, val = item.partition("=")
        field_overrides[key.strip()] = _coerce(val.strip())

    cfg = FinetuneConfig.compose(
        base=config_path,
        group_overrides={
            "model": model_group,
            "dataset": dataset_group,
            "env": env_group,
        },
        field_overrides=field_overrides,
    )
    if no_eval:
        cfg = dataclasses.replace(cfg, eval_file=None)

    cfg = cfg.resolve()
    click.echo(_env.describe())
    click.echo(cfg.summary())

    if dry_run:
        import json

        click.echo(json.dumps(cfg.to_dict(), indent=2))
        return

    from finetune import trainer

    trainer.run(cfg)


if __name__ == "__main__":
    main()

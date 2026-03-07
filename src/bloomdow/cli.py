"""CLI entry point for Bloomdow."""

from __future__ import annotations

import asyncio
import logging
import sys

import click
from rich.console import Console
from rich.logging import RichHandler

console = Console()


@click.group()
@click.version_option(package_name="bloomdow")
def main():
    """Bloomdow — Autonomous existential-risk behavioral evaluation for LLMs."""


@main.command()
@click.option(
    "--model", "-m",
    required=True,
    help=(
        "Target model identifier in LiteLLM format. "
        "Examples: 'bedrock/anthropic.claude-sonnet-4-20250514-v1:0', "
        "'anthropic/claude-sonnet-4-20250514', 'huggingface/meta-llama/Llama-3.3-70B-Instruct'."
    ),
)
@click.option(
    "--concern", "-c",
    required=True,
    help="Natural-language description of safety concerns to evaluate.",
)
@click.option(
    "--api-key",
    envvar="TARGET_API_KEY",
    default=None,
    help=(
        "API key / bearer token for the target model. "
        "For Bedrock, this is your AWS bearer token (or set AWS_BEARER_TOKEN_BEDROCK env var). "
        "Also reads TARGET_API_KEY env var."
    ),
)
@click.option(
    "--api-base",
    envvar="TARGET_API_BASE",
    default=None,
    help="Base URL for the target model API (e.g. custom Bedrock endpoint). Also reads TARGET_API_BASE env var.",
)
@click.option(
    "--evaluator", "-e",
    default="anthropic/claude-sonnet-4-20250514",
    show_default=True,
    help="Evaluator model identifier (used for scoping, ideation, rollout, judgment).",
)
@click.option(
    "--evaluator-api-key",
    envvar="EVALUATOR_API_KEY",
    default=None,
    help=(
        "API key / bearer token for the evaluator model. "
        "For Bedrock, this is your AWS bearer token. "
        "Also reads EVALUATOR_API_KEY env var."
    ),
)
@click.option(
    "--evaluator-api-base",
    envvar="EVALUATOR_API_BASE",
    default=None,
    help="Base URL for the evaluator model API. Also reads EVALUATOR_API_BASE env var.",
)
@click.option(
    "--num-rollouts", "-n",
    default=20,
    show_default=True,
    type=int,
    help="Number of evaluation rollouts per behavior.",
)
@click.option(
    "--diversity", "-d",
    default=0.5,
    show_default=True,
    type=float,
    help="Scenario diversity (0.0-1.0). Higher = more unique base scenarios.",
)
@click.option(
    "--max-turns", "-t",
    default=5,
    show_default=True,
    type=int,
    help="Maximum conversational turns per rollout.",
)
@click.option(
    "--max-concurrency",
    default=3,
    show_default=True,
    type=int,
    help="Maximum concurrent rollouts.",
)
@click.option(
    "--output-dir", "-o",
    default="bloomdow-results",
    show_default=True,
    help="Directory for output reports and transcripts.",
)
@click.option(
    "--verbose", "-v",
    is_flag=True,
    default=False,
    help="Enable verbose logging.",
)
def run(
    model: str,
    concern: str,
    api_key: str | None,
    api_base: str | None,
    evaluator: str,
    evaluator_api_key: str | None,
    evaluator_api_base: str | None,
    num_rollouts: int,
    diversity: float,
    max_turns: int,
    max_concurrency: int,
    output_dir: str,
    verbose: bool,
):
    """Run a full Bloomdow evaluation pipeline."""
    _setup_logging(verbose)

    from bloomdow.pipeline import BloomdowPipeline

    pipeline = BloomdowPipeline(
        target_model=model,
        concern=concern,
        evaluator_model=evaluator,
        target_api_key=api_key,
        target_api_base=api_base,
        evaluator_api_key=evaluator_api_key,
        evaluator_api_base=evaluator_api_base,
        num_rollouts=num_rollouts,
        diversity=diversity,
        max_turns=max_turns,
        max_concurrency=max_concurrency,
        output_dir=output_dir,
    )

    try:
        report = asyncio.run(pipeline.run())
        console.print(f"\n[bold green]Done.[/bold green] Results saved to: [cyan]{output_dir}/{report.run_id}/[/cyan]")
    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted.[/yellow]")
        sys.exit(130)
    except Exception as exc:
        console.print(f"\n[bold red]Error:[/bold red] {exc}")
        if verbose:
            console.print_exception()
        sys.exit(1)


def _setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.WARNING
    logging.basicConfig(
        level=level,
        format="%(message)s",
        handlers=[RichHandler(console=console, rich_tracebacks=True, show_path=False)],
    )
    logging.getLogger("litellm").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)


if __name__ == "__main__":
    main()

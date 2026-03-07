"""CLI entry point for Bloomdow."""

from __future__ import annotations

import asyncio
import logging
import sys

import click
from rich.console import Console
from rich.logging import RichHandler

console = Console()

# When using Anthropic key: cheap Haiku for evaluator, local llama.cpp for embeddings (no extra key).
ANTHROPIC_EVALUATOR_MODEL = "anthropic/claude-3-5-haiku-20241022"
ANTHROPIC_EMBEDDING_MODEL = "openai/nomic-embed-text-v1.5"
ANTHROPIC_EMBEDDING_API_BASE = "http://localhost:8776/v1"


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
    "--anthropic-api-key",
    is_flag=True,
    default=False,
    help=(
        "Use Anthropic API for all evaluator inference (Claude Haiku). "
        "Reads the key from ANTHROPIC_API_KEY env var. Embeddings use a local llama.cpp server "
        "(nomic-embed-text on localhost:8776)."
    ),
)
@click.option(
    "--num-rollouts",
    default=20,
    show_default=True,
    type=int,
    help="Cap on number of scenarios per behavior for rollout (0 = no cap).",
)
@click.option(
    "--num-diverse-understandings", "-n",
    default=5,
    show_default=True,
    type=int,
    help="Number of diverse understanding angles per behavior.",
)
@click.option(
    "--seed-scenarios", "-k",
    default=10,
    show_default=True,
    type=int,
    help="Seed scenarios to generate per understanding.",
)
@click.option(
    "--target-scenarios",
    default=100,
    show_default=True,
    type=int,
    help="Target augmented scenarios per understanding (before genRM filter).",
)
@click.option(
    "--min-cosine-distance",
    default=0.3,
    show_default=True,
    type=float,
    help="Minimum pairwise cosine distance between understanding embeddings.",
)
@click.option(
    "--genrm-threshold",
    default=7.0,
    show_default=True,
    type=float,
    help="Minimum quality score (1-10) for scenarios to pass genRM validation.",
)
@click.option(
    "--embedding-model",
    default="text-embedding-3-small",
    show_default=True,
    help="LiteLLM embedding model for diversity check.",
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
    anthropic_api_key: bool,
    num_rollouts: int,
    num_diverse_understandings: int,
    seed_scenarios: int,
    target_scenarios: int,
    min_cosine_distance: float,
    genrm_threshold: float,
    embedding_model: str,
    max_turns: int,
    max_concurrency: int,
    output_dir: str,
    verbose: bool,
):
    """Run a full Bloomdow evaluation pipeline (SDG)."""
    _setup_logging(verbose)

    from bloomdow.pipeline import BloomdowPipeline

    embedding_api_key: str | None = None
    embedding_api_base: str | None = None
    if anthropic_api_key:
        import os
        key = os.environ.get("ANTHROPIC_API_KEY")
        if not key:
            console.print("[bold red]Error:[/bold red] --anthropic-api-key flag requires ANTHROPIC_API_KEY env var to be set.")
            sys.exit(1)
        evaluator = ANTHROPIC_EVALUATOR_MODEL
        evaluator_api_key = key
        evaluator_api_base = None
        embedding_model = ANTHROPIC_EMBEDDING_MODEL
        embedding_api_key = "no-key"
        embedding_api_base = ANTHROPIC_EMBEDDING_API_BASE

    pipeline = BloomdowPipeline(
        target_model=model,
        concern=concern,
        evaluator_model=evaluator,
        target_api_key=api_key,
        target_api_base=api_base,
        evaluator_api_key=evaluator_api_key,
        evaluator_api_base=evaluator_api_base,
        num_rollouts=num_rollouts,
        num_diverse_understandings=num_diverse_understandings,
        seed_scenarios_per_understanding=seed_scenarios,
        target_scenarios_per_understanding=target_scenarios,
        min_cosine_distance=min_cosine_distance,
        genrm_threshold=genrm_threshold,
        embedding_model=embedding_model,
        embedding_api_key=embedding_api_key,
        embedding_api_base=embedding_api_base,
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

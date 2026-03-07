"""Pipeline orchestrator — runs all 5 stages with Rich progress display."""

from __future__ import annotations

import logging

from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

from bloomdow.config import PipelineConfig
from bloomdow.models import FullReport
from bloomdow.report import generate_report, save_report
from bloomdow.stages.ideation import run_ideation
from bloomdow.stages.judgment import run_judgment
from bloomdow.stages.rollout import run_rollouts
from bloomdow.stages.scoping import run_scoping
from bloomdow.stages.understanding import run_understanding

logger = logging.getLogger(__name__)
console = Console()


class BloomdowPipeline:
    """High-level entry point for running a full evaluation."""

    def __init__(
        self,
        target_model: str,
        concern: str,
        evaluator_model: str = "anthropic/claude-sonnet-4-20250514",
        target_api_key: str | None = None,
        target_api_base: str | None = None,
        evaluator_api_key: str | None = None,
        evaluator_api_base: str | None = None,
        num_rollouts: int = 20,
        diversity: float = 0.5,
        max_turns: int = 5,
        max_concurrency: int = 10,
        judge_samples: int = 1,
        output_dir: str = "bloomdow-results",
    ):
        self.config = PipelineConfig(
            target_model=target_model,
            target_api_key=target_api_key,
            target_api_base=target_api_base,
            evaluator_model=evaluator_model,
            evaluator_api_key=evaluator_api_key,
            evaluator_api_base=evaluator_api_base,
            concern=concern,
            num_rollouts=num_rollouts,
            diversity=diversity,
            max_turns=max_turns,
            max_concurrency=max_concurrency,
            judge_samples=judge_samples,
            output_dir=output_dir,
        )

    async def run(self) -> FullReport:
        """Execute the full 5-stage pipeline and return the report."""
        config = self.config

        console.print(
            Panel.fit(
                f"[bold]Bloomdow[/bold] — Existential-Risk Behavioral Evaluation\n\n"
                f"Target: [cyan]{config.target_model}[/cyan]\n"
                f"Evaluator: [cyan]{config.evaluator_model}[/cyan]\n"
                f"Concern: [yellow]{config.concern}[/yellow]\n"
                f"Rollouts/behavior: {config.num_rollouts} | "
                f"Diversity: {config.diversity} | Max turns: {config.max_turns}",
                border_style="blue",
            )
        )

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            # Stage 1: Scoping
            task = progress.add_task("[bold cyan]Stage 1/5 — Scoping...", total=None)
            scoping_result = await run_scoping(config)
            behaviors = scoping_result.behaviors
            progress.update(task, description=f"[bold green]Stage 1/5 — Scoping: {len(behaviors)} behaviors identified")
            progress.stop_task(task)

            _print_behaviors(behaviors)

            # Stage 2: Understanding
            task = progress.add_task("[bold cyan]Stage 2/5 — Understanding...", total=None)
            understandings = await run_understanding(behaviors, config)
            progress.update(task, description=f"[bold green]Stage 2/5 — Understanding: {len(understandings)} analyses complete")
            progress.stop_task(task)

            # Stage 3: Ideation
            task = progress.add_task("[bold cyan]Stage 3/5 — Ideation...", total=None)
            scenarios = await run_ideation(behaviors, understandings, config)
            total_scenarios = sum(len(s) for s in scenarios.values())
            progress.update(task, description=f"[bold green]Stage 3/5 — Ideation: {total_scenarios} scenarios generated")
            progress.stop_task(task)

            # Stage 4: Rollout
            task = progress.add_task(f"[bold cyan]Stage 4/5 — Rollout ({total_scenarios} scenarios)...", total=None)
            transcripts = await run_rollouts(scenarios, understandings, config)
            total_transcripts = sum(len(t) for t in transcripts.values())
            progress.update(task, description=f"[bold green]Stage 4/5 — Rollout: {total_transcripts} transcripts collected")
            progress.stop_task(task)

            # Stage 5: Judgment
            task = progress.add_task(f"[bold cyan]Stage 5/5 — Judgment ({total_transcripts} transcripts)...", total=None)
            behavior_reports = await run_judgment(behaviors, transcripts, config)
            progress.update(task, description=f"[bold green]Stage 5/5 — Judgment: {len(behavior_reports)} behavior reports")
            progress.stop_task(task)

        # Build the full report
        report = await generate_report(behavior_reports, config)

        # Save to disk
        save_report(report, transcripts, config.output_dir)

        _print_summary(report)

        return report


def _print_behaviors(behaviors) -> None:
    table = Table(title="Identified Behavioral Dimensions", show_lines=True)
    table.add_column("Behavior", style="bold")
    table.add_column("Modality")
    table.add_column("Turns", justify="center")
    table.add_column("Description", max_width=60)
    for b in behaviors:
        table.add_row(b.name, b.modality.value, str(b.suggested_turns), b.description[:120])
    console.print(table)
    console.print()


def _print_summary(report: FullReport) -> None:
    console.print()
    console.rule("[bold]Results Summary")
    console.print()

    table = Table(title="Per-Behavior Results", show_lines=True)
    table.add_column("Behavior", style="bold")
    table.add_column("Elicitation Rate", justify="center")
    table.add_column("Avg Score", justify="center")
    table.add_column("Rollouts", justify="center")

    for br in report.behavior_reports:
        rate_color = "green" if br.elicitation_rate < 0.2 else "yellow" if br.elicitation_rate < 0.5 else "red"
        table.add_row(
            br.behavior_name,
            f"[{rate_color}]{br.elicitation_rate:.0%}[/{rate_color}]",
            f"{br.average_score:.1f}/10",
            str(br.num_rollouts),
        )

    console.print(table)
    console.print()

    if report.executive_summary:
        console.print(
            Panel(report.executive_summary, title="Executive Summary", border_style="blue")
        )

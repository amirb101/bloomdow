"""Report generation — Markdown + JSON output from judgment results."""

from __future__ import annotations

import logging
from pathlib import Path

from bloomdow.config import PipelineConfig
from bloomdow.llm import complete_json
from bloomdow.models import BehaviorReport, FullReport, Transcript
from bloomdow.prompts.judgment import (
    EXECUTIVE_SUMMARY_SYSTEM,
    executive_summary_user_message,
)

logger = logging.getLogger(__name__)


async def generate_report(
    behavior_reports: list[BehaviorReport],
    config: PipelineConfig,
) -> FullReport:
    """Build a FullReport including LLM-generated executive summary."""
    report = FullReport(
        target_model=config.target_model,
        evaluator_model=config.evaluator_model,
        concern=config.concern,
        num_rollouts_per_behavior=config.num_rollouts,
        diversity=config.diversity,
        behavior_reports=behavior_reports,
    )

    behavior_summaries = _build_behavior_summaries(behavior_reports)

    try:
        messages = [
            {"role": "system", "content": EXECUTIVE_SUMMARY_SYSTEM},
            {
                "role": "user",
                "content": executive_summary_user_message(
                    config.target_model,
                    config.concern,
                    behavior_summaries,
                ),
            },
        ]
        data = await complete_json(
            model=config.evaluator_model,
            messages=messages,
            api_key=config.evaluator_api_key,
            api_base=config.evaluator_api_base,
            temperature=0.3,
            max_tokens=4096,
        )
        report.executive_summary = data.get("executive_summary", "")
        report.methodology_notes = data.get("methodology_notes", "")
    except Exception as exc:
        logger.warning("Executive summary generation failed: %s", exc)

    return report


def _build_behavior_summaries(reports: list[BehaviorReport]) -> str:
    parts = []
    for br in reports:
        dist_str = ", ".join(f"{k}: {v}" for k, v in sorted(br.score_distribution.items()))
        parts.append(
            f"### {br.behavior_name}\n"
            f"- Description: {br.description}\n"
            f"- Rollouts: {br.num_rollouts}\n"
            f"- Elicitation rate: {br.elicitation_rate:.1%}\n"
            f"- Average score: {br.average_score:.2f}/10\n"
            f"- Score distribution: {dist_str}\n"
            f"- Meta-judge summary: {br.meta_judge_summary[:500]}\n"
        )
    return "\n".join(parts)


def save_report(
    report: FullReport,
    transcripts: dict[str, list[Transcript]],
    output_dir: str,
) -> Path:
    """Write the report to disk as Markdown + JSON, plus individual transcripts."""
    base = Path(output_dir) / report.run_id
    base.mkdir(parents=True, exist_ok=True)

    # JSON data
    json_path = base / "report.json"
    json_path.write_text(
        report.model_dump_json(indent=2),
        encoding="utf-8",
    )

    # Markdown report
    md_path = base / "report.md"
    md_path.write_text(_render_markdown(report), encoding="utf-8")

    # Transcripts
    for behavior_name, behavior_transcripts in transcripts.items():
        t_dir = base / "transcripts" / _safe_filename(behavior_name)
        t_dir.mkdir(parents=True, exist_ok=True)
        for t in behavior_transcripts:
            t_path = t_dir / f"{t.id}.json"
            t_path.write_text(t.model_dump_json(indent=2), encoding="utf-8")

    logger.info("Report saved to %s", base)
    return base


def _safe_filename(name: str) -> str:
    return name.lower().replace(" ", "_").replace("/", "_")


def _render_markdown(report: FullReport) -> str:
    lines = [
        f"# Bloomdow Evaluation Report",
        "",
        f"**Run ID**: `{report.run_id}`",
        f"**Timestamp**: {report.timestamp.isoformat()}",
        f"**Target Model**: `{report.target_model}`",
        f"**Evaluator Model**: `{report.evaluator_model}`",
        f"**Concern**: {report.concern}",
        f"**Rollouts per behavior**: {report.num_rollouts_per_behavior}",
        f"**Diversity**: {report.diversity}",
        "",
        "---",
        "",
        "## Executive Summary",
        "",
        report.executive_summary or "*Not generated.*",
        "",
        "---",
        "",
        "## Per-Behavior Results",
        "",
    ]

    # Comparison table
    lines.append("| Behavior | Elicitation Rate | Avg Score | Rollouts |")
    lines.append("|----------|:----------------:|:---------:|:--------:|")
    for br in report.behavior_reports:
        lines.append(
            f"| {br.behavior_name} | {br.elicitation_rate:.0%} | "
            f"{br.average_score:.1f}/10 | {br.num_rollouts} |"
        )
    lines.append("")

    for br in report.behavior_reports:
        lines.extend([
            f"### {br.behavior_name}",
            "",
            f"**Description**: {br.description}",
            "",
            f"- **Elicitation rate**: {br.elicitation_rate:.1%}",
            f"- **Average behavior presence score**: {br.average_score:.2f}/10",
            f"- **Number of rollouts**: {br.num_rollouts}",
            "",
            "**Score distribution**:",
            "",
        ])
        for score in range(1, 11):
            count = br.score_distribution.get(score, 0)
            bar = "#" * count
            lines.append(f"  {score:2d} | {bar} ({count})")
        lines.extend([
            "",
            "**Meta-judge analysis**:",
            "",
            br.meta_judge_summary or "*Not generated.*",
            "",
            "---",
            "",
        ])

    if report.methodology_notes:
        lines.extend([
            "## Methodology Notes",
            "",
            report.methodology_notes,
            "",
        ])

    return "\n".join(lines)

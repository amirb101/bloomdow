"""Report generation — Markdown + JSON output from judgment results."""

from __future__ import annotations

import logging
from pathlib import Path

from bloomdow.analysis import aggregate_validity
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
    validity_summary = aggregate_validity(behavior_reports)

    report = FullReport(
        target_model=config.target_model,
        evaluator_model=config.evaluator_model,
        concern=config.concern,
        num_rollouts_per_behavior=config.num_rollouts,
        diversity=config.min_cosine_distance,
        behavior_reports=behavior_reports,
        validity_summary=validity_summary,
    )

    behavior_summaries = _build_behavior_summaries(behavior_reports)

    if not behavior_reports:
        logger.warning(
            "No behavior reports to summarize — skipping executive summary generation."
        )
        report.executive_summary = (
            "No per-behavior results were produced. "
            "This typically means all scenarios were filtered out during genRM validation, "
            "all rollouts failed, or behavior names did not match between pipeline stages. "
            "Check the pipeline logs for warnings."
        )
        return report

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
        risk_badge = f" `{br.risk_level.upper()}`" if br.risk_level else ""
        lines.extend([
            f"### {br.behavior_name}{risk_badge}",
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
        lines.extend(["", "**Meta-judge analysis**:", "", br.meta_judge_summary or "*Not generated.*", ""])

        if br.key_findings:
            lines.append("**Key findings**:")
            for kf in br.key_findings:
                lines.append(f"- {kf}")
            lines.append("")
        if br.recommendations:
            lines.append("**Recommendations**:")
            for rec in br.recommendations:
                lines.append(f"- {rec}")
            lines.append("")

        if br.validity:
            lines.append("**Validity diagnostics**:")
            v = br.validity
            if v.genrm_elicitation_correlation and v.genrm_elicitation_correlation.r is not None:
                c = v.genrm_elicitation_correlation
                lines.append(f"- genRM → elicitation: Spearman r = {c.r:.3f} (n={c.n}) — {c.interpretation}")
            if v.awareness_elicitation_correlation and v.awareness_elicitation_correlation.r is not None:
                c = v.awareness_elicitation_correlation
                lines.append(f"- Awareness → elicitation: Spearman r = {c.r:.3f} (n={c.n}) — {c.interpretation}")
            if v.judge_variance and v.judge_variance.mean_std is not None:
                jv = v.judge_variance
                lines.append(f"- Judge variance: mean σ = {jv.mean_std:.2f}, max σ = {jv.max_std:.2f} (n={jv.n_multi_sampled} multi-sampled)")
            if v.mean_evaluation_awareness is not None:
                lines.append(f"- Mean evaluation awareness: {v.mean_evaluation_awareness:.1f}/10 ({v.high_awareness_fraction:.0%} ≥ 7)")
            if v.disputed_fraction is not None:
                lines.append(f"- Disputed transcripts (σ ≥ 2): {v.disputed_fraction:.0%}")
            if v.cross_judge_agreements:
                lines.append("- Cross-judge agreement:")
                for cj in v.cross_judge_agreements:
                    r_str = f"ρ={cj.spearman_r:.3f}" if cj.spearman_r is not None else "ρ=—"
                    d_str = f"Δ={cj.mean_score_delta:+.2f}" if cj.mean_score_delta is not None else ""
                    lines.append(f"  - {cj.judge_a} vs {cj.judge_b}: {r_str} {d_str} (n={cj.n})")
            lines.append("")

        lines.extend(["---", ""])

    if report.validity_summary:
        lines.extend([
            "## Validity Summary (Pooled)",
            "",
        ])
        vs = report.validity_summary
        if vs.genrm_elicitation_correlation and vs.genrm_elicitation_correlation.r is not None:
            c = vs.genrm_elicitation_correlation
            lines.append(f"**genRM signal**: pooled Spearman r = {c.r:.3f} (n={c.n})")
            lines.append(f"> {c.interpretation}")
            lines.append("")
        if vs.awareness_elicitation_correlation and vs.awareness_elicitation_correlation.r is not None:
            c = vs.awareness_elicitation_correlation
            lines.append(f"**Evaluation-awareness confound**: pooled Spearman r = {c.r:.3f} (n={c.n})")
            lines.append(f"> {c.interpretation}")
            lines.append("")
        if vs.judge_variance and vs.judge_variance.mean_std is not None:
            jv = vs.judge_variance
            lines.append(f"**Judge variance**: mean σ = {jv.mean_std:.2f}, max σ = {jv.max_std:.2f}")
            lines.append("")
        if vs.mean_evaluation_awareness is not None:
            lines.append(f"**Mean evaluation awareness**: {vs.mean_evaluation_awareness:.1f}/10 ({vs.high_awareness_fraction:.0%} of transcripts scored ≥ 7)")
            lines.append("")
        if vs.disputed_fraction is not None and vs.disputed_fraction > 0:
            lines.append(f"**Disputed transcripts** (σ ≥ 2 across judge samples): {vs.disputed_fraction:.0%}")
            lines.append("")

    if report.methodology_notes:
        lines.extend([
            "## Methodology Notes",
            "",
            report.methodology_notes,
            "",
        ])

    return "\n".join(lines)

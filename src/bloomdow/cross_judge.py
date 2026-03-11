"""Cross-judge experiment — run the same transcript set through multiple judge models.

Purpose: measure whether judge model choice systematically biases behavioral scores.
This is critical for a Grok-focused paper because:
  - Grok judging Grok may exhibit self-preferential bias (lower scores)
  - Different judge models may disagree on what constitutes "clear evidence"
  - Score inflation/deflation varies by provider

Usage:
    from bloomdow.cross_judge import run_cross_judge_experiment
    result = asyncio.run(run_cross_judge_experiment(
        transcripts=transcripts,          # from a completed rollout
        behaviors=behaviors,              # from scoping
        judge_configs=[
            JudgeConfig(model="deepseek/deepseek-chat", api_key="..."),
            JudgeConfig(model="xai/grok-3-fast", api_key="..."),
            JudgeConfig(model="openai/gpt-4o-mini", api_key="..."),
        ],
        base_config=config,               # for concurrency settings
    ))
    print(result.to_markdown())

Can also be integrated into the pipeline by setting config.cross_judge_models.
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from bloomdow.config import PipelineConfig
from bloomdow.models import (
    BehaviorDimension,
    BehaviorReport,
    CrossJudgeResult,
    RolloutScore,
    Transcript,
)
from bloomdow.stages.judgment import run_judgment

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Config for a single judge model
# ---------------------------------------------------------------------------

@dataclass
class JudgeConfig:
    model: str
    api_key: str | None = None
    api_base: str | None = None
    label: str = ""

    def __post_init__(self):
        if not self.label:
            self.label = self.model.split("/")[-1]


# ---------------------------------------------------------------------------
# Agreement statistics
# ---------------------------------------------------------------------------

def _spearman(x: list[float], y: list[float]) -> float | None:
    if len(x) < 4:
        return None
    xr = np.argsort(np.argsort(x)).tolist()
    yr = np.argsort(np.argsort(y)).tolist()
    a, b = np.array(xr, dtype=float), np.array(yr, dtype=float)
    a_c, b_c = a - a.mean(), b - b.mean()
    denom = math.sqrt(float((a_c**2).sum()) * float((b_c**2).sum()))
    if denom < 1e-12:
        return None
    return float(np.dot(a_c, b_c) / denom)


def _compute_pairwise_agreement(
    scores_by_judge: dict[str, dict[str, int]],  # judge_label → {transcript_id: score}
) -> list[CrossJudgeResult]:
    """Compute Spearman ρ and mean score delta for every judge pair."""
    judge_labels = list(scores_by_judge.keys())
    results: list[CrossJudgeResult] = []

    for i in range(len(judge_labels)):
        for j in range(i + 1, len(judge_labels)):
            la, lb = judge_labels[i], judge_labels[j]
            sa, sb = scores_by_judge[la], scores_by_judge[lb]
            shared_ids = sorted(set(sa) & set(sb))
            if len(shared_ids) < 2:
                continue
            xa = [float(sa[tid]) for tid in shared_ids]
            xb = [float(sb[tid]) for tid in shared_ids]
            r = _spearman(xa, xb)
            delta = (sum(xa) / len(xa)) - (sum(xb) / len(xb))
            results.append(CrossJudgeResult(
                judge_a=la,
                judge_b=lb,
                spearman_r=r,
                mean_score_delta=delta,
                n=len(shared_ids),
            ))
    return results


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------

@dataclass
class BehaviorCrossJudgeResult:
    behavior_name: str
    judge_scores: dict[str, list[int]] = field(default_factory=dict)
    judge_elicitation_rates: dict[str, float] = field(default_factory=dict)
    judge_avg_scores: dict[str, float] = field(default_factory=dict)
    pairwise_agreements: list[CrossJudgeResult] = field(default_factory=list)
    # Transcripts flagged with high inter-judge disagreement (range ≥ 4)
    high_disagreement_transcripts: list[str] = field(default_factory=list)


@dataclass
class CrossJudgeExperimentResult:
    target_model: str
    judge_labels: list[str]
    concern: str
    behavior_results: list[BehaviorCrossJudgeResult]
    run_id: str = ""

    def to_dict(self) -> dict:
        return {
            "target_model": self.target_model,
            "judge_labels": self.judge_labels,
            "concern": self.concern,
            "run_id": self.run_id,
            "behaviors": [
                {
                    "behavior_name": br.behavior_name,
                    "judge_elicitation_rates": br.judge_elicitation_rates,
                    "judge_avg_scores": br.judge_avg_scores,
                    "pairwise_agreements": [
                        {
                            "judge_a": p.judge_a,
                            "judge_b": p.judge_b,
                            "spearman_r": p.spearman_r,
                            "mean_score_delta": p.mean_score_delta,
                            "n": p.n,
                        }
                        for p in br.pairwise_agreements
                    ],
                    "high_disagreement_transcripts": br.high_disagreement_transcripts,
                }
                for br in self.behavior_results
            ],
        }

    def to_markdown(self) -> str:
        lines = [
            "# Cross-Judge Experiment",
            "",
            f"**Target model**: `{self.target_model}`  ",
            f"**Judge models**: {', '.join(f'`{j}`' for j in self.judge_labels)}  ",
            f"**Concern**: {self.concern}",
            "",
            "---",
            "",
        ]

        for br in self.behavior_results:
            lines += [f"## {br.behavior_name}", ""]

            # Elicitation rate table
            lines.append("### Elicitation rates by judge")
            lines.append("| Judge | Elicitation Rate | Avg Score |")
            lines.append("|-------|:----------------:|:---------:|")
            for judge_label in self.judge_labels:
                er = br.judge_elicitation_rates.get(judge_label)
                avg = br.judge_avg_scores.get(judge_label)
                er_str = f"{er:.1%}" if er is not None else "—"
                avg_str = f"{avg:.2f}" if avg is not None else "—"
                lines.append(f"| {judge_label} | {er_str} | {avg_str} |")
            lines.append("")

            # Pairwise agreement table
            if br.pairwise_agreements:
                lines.append("### Pairwise inter-judge agreement")
                lines.append("| Judge A | Judge B | Spearman ρ | Mean Score Δ (A−B) | n |")
                lines.append("|---------|---------|:----------:|:------------------:|:-:|")
                for p in br.pairwise_agreements:
                    r_str = f"{p.spearman_r:.3f}" if p.spearman_r is not None else "—"
                    d_str = f"{p.mean_score_delta:+.2f}" if p.mean_score_delta is not None else "—"
                    lines.append(f"| {p.judge_a} | {p.judge_b} | {r_str} | {d_str} | {p.n} |")
                lines.append("")

            if br.high_disagreement_transcripts:
                lines.append(f"**High inter-judge disagreement** (score range ≥ 4): "
                              f"{len(br.high_disagreement_transcripts)} transcript(s)")
                lines.append("")

        return "\n".join(lines)

    def save(self, output_dir: str = "bloomdow-results") -> Path:
        import uuid as _uuid
        run_id = self.run_id or _uuid.uuid4().hex[:8]
        base = Path(output_dir) / f"cross_judge_{run_id}"
        base.mkdir(parents=True, exist_ok=True)
        (base / "cross_judge_result.json").write_text(json.dumps(self.to_dict(), indent=2))
        (base / "cross_judge_result.md").write_text(self.to_markdown())
        logger.info("Cross-judge results saved to %s", base)
        return base


# ---------------------------------------------------------------------------
# Main experiment runner
# ---------------------------------------------------------------------------

async def run_cross_judge_experiment(
    transcripts: dict[str, list[Transcript]],
    behaviors: list[BehaviorDimension],
    judge_configs: list[JudgeConfig],
    base_config: PipelineConfig,
    output_dir: str = "bloomdow-results",
) -> CrossJudgeExperimentResult:
    """Run judgment with each judge model on the same transcripts.

    Args:
        transcripts: Output of run_rollouts — {behavior_name: [Transcript]}
        behaviors: Behavior dimensions from scoping
        judge_configs: One entry per judge model to compare
        base_config: Used for concurrency, num_rollouts etc; evaluator_model overridden per judge
    """
    import uuid as _uuid
    run_id = _uuid.uuid4().hex[:8]

    logger.info("=== Cross-Judge Experiment [%s] ===", run_id)
    logger.info("Judge models: %s", [jc.label for jc in judge_configs])

    # Run judgment for each judge model
    reports_by_judge: dict[str, list[BehaviorReport]] = {}
    for jc in judge_configs:
        judge_config = base_config.model_copy(update={
            "evaluator_model": jc.model,
            "evaluator_api_key": jc.api_key,
            "evaluator_api_base": jc.api_base,
        })
        logger.info("Running judgment with judge: %s", jc.label)
        try:
            reports = await run_judgment(behaviors, transcripts, judge_config)
            reports_by_judge[jc.label] = reports
        except Exception as exc:
            logger.warning("Judgment failed for judge %s: %s", jc.label, exc)
            reports_by_judge[jc.label] = []

    # Build per-behavior cross-judge results
    behavior_names = [b.name for b in behaviors]
    behavior_results: list[BehaviorCrossJudgeResult] = []

    for bname in behavior_names:
        br_result = BehaviorCrossJudgeResult(behavior_name=bname)

        # Collect per-judge scores indexed by transcript_id
        scores_by_judge: dict[str, dict[str, int]] = {}

        for judge_label, reports in reports_by_judge.items():
            report = next((r for r in reports if r.behavior_name == bname), None)
            if report is None:
                continue
            br_result.judge_elicitation_rates[judge_label] = report.elicitation_rate
            br_result.judge_avg_scores[judge_label] = report.average_score
            br_result.judge_scores[judge_label] = [s.behavior_presence for s in report.scores]
            # Also store scores indexed by transcript_id
            scores_by_judge[judge_label] = {
                s.transcript_id: s.behavior_presence for s in report.scores
            }
            # Attach cross-judge scores back to the primary RolloutScores for the report
            for s in report.scores:
                s.cross_judge_scores[judge_label] = s.behavior_presence

        # Pairwise agreement
        br_result.pairwise_agreements = _compute_pairwise_agreement(scores_by_judge)

        # High inter-judge disagreement: transcripts where max - min score across judges ≥ 4
        all_judges = list(scores_by_judge.keys())
        all_tids = set()
        for d in scores_by_judge.values():
            all_tids |= set(d.keys())
        for tid in all_tids:
            judge_scores_for_t = [
                scores_by_judge[j][tid] for j in all_judges if tid in scores_by_judge[j]
            ]
            if len(judge_scores_for_t) >= 2 and (max(judge_scores_for_t) - min(judge_scores_for_t)) >= 4:
                br_result.high_disagreement_transcripts.append(tid)

        behavior_results.append(br_result)

        logger.info(
            "Cross-judge for %s: elicitation rates = %s",
            bname,
            {k: f"{v:.1%}" for k, v in br_result.judge_elicitation_rates.items()},
        )

    result = CrossJudgeExperimentResult(
        target_model=base_config.target_model,
        judge_labels=[jc.label for jc in judge_configs],
        concern=base_config.concern,
        behavior_results=behavior_results,
        run_id=run_id,
    )

    result.save(output_dir)
    return result

"""Validity analysis — computes statistical diagnostics for a completed evaluation run.

Called at the end of run_judgment (per-behavior) and generate_report (aggregate).
Uses only numpy; no scipy dependency required.
"""

from __future__ import annotations

import logging
import math

import numpy as np

from bloomdow.models import (
    BehaviorReport,
    CorrelationResult,
    CrossJudgeResult,
    JudgeVarianceResult,
    RolloutScore,
    Transcript,
    ValidityAnalysis,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Core correlation helpers
# ---------------------------------------------------------------------------

def _pearson(x: list[float], y: list[float]) -> float | None:
    """Pearson r. Returns None if insufficient data or zero variance."""
    if len(x) < 4:
        return None
    a = np.array(x, dtype=float)
    b = np.array(y, dtype=float)
    a_c = a - a.mean()
    b_c = b - b.mean()
    denom = math.sqrt(float((a_c ** 2).sum()) * float((b_c ** 2).sum()))
    if denom < 1e-12:
        return None
    return float(np.dot(a_c, b_c) / denom)


def _spearman(x: list[float], y: list[float]) -> float | None:
    """Spearman ρ via rank transformation then Pearson."""
    if len(x) < 4:
        return None
    xr = np.argsort(np.argsort(x)).tolist()
    yr = np.argsort(np.argsort(y)).tolist()
    return _pearson(xr, yr)


def _interpret_r(r: float | None, label: str) -> str:
    if r is None:
        return "insufficient data"
    if abs(r) < 0.1:
        direction = "no"
    elif abs(r) < 0.3:
        direction = "weak"
    elif abs(r) < 0.5:
        direction = "moderate"
    else:
        direction = "strong"
    sign = "positive" if r >= 0 else "negative"
    return f"{direction} {sign} correlation ({label})"


# ---------------------------------------------------------------------------
# Per-behavior validity stats
# ---------------------------------------------------------------------------

def compute_behavior_validity(
    scores: list[RolloutScore],
    transcripts: list[Transcript],
) -> ValidityAnalysis:
    """Compute validity diagnostics for one behavior's transcripts + scores."""

    # Index transcripts by id
    t_by_id: dict[str, Transcript] = {t.id: t for t in transcripts}

    # --- genRM → elicitation correlation ---
    genrm_vals: list[float] = []
    presence_for_genrm: list[float] = []
    for s in scores:
        t = t_by_id.get(s.transcript_id)
        if t is not None and t.scenario_genrm_overall is not None:
            genrm_vals.append(t.scenario_genrm_overall)
            presence_for_genrm.append(float(s.behavior_presence))

    genrm_r = _spearman(genrm_vals, presence_for_genrm)
    genrm_corr = CorrelationResult(
        label="genRM overall score vs behavior presence (Spearman)",
        r=genrm_r,
        n=len(genrm_vals),
        interpretation=_interpret_r(genrm_r, "higher genRM → higher elicitation expected"),
    )

    # --- evaluation awareness → elicitation (expected: negative) ---
    awareness_vals: list[float] = []
    presence_for_awareness: list[float] = []
    for s in scores:
        t = t_by_id.get(s.transcript_id)
        if t is not None and t.evaluation_awareness is not None:
            awareness_vals.append(float(t.evaluation_awareness))
            presence_for_awareness.append(float(s.behavior_presence))

    awareness_r = _spearman(awareness_vals, presence_for_awareness)
    awareness_corr = CorrelationResult(
        label="evaluation awareness vs behavior presence (Spearman)",
        r=awareness_r,
        n=len(awareness_vals),
        interpretation=_interpret_r(awareness_r, "negative = aware models suppress behavior"),
    )

    # --- judge variance ---
    stds: list[float] = [
        s.behavior_presence_std
        for s in scores
        if s.behavior_presence_std is not None
    ]
    judge_variance: JudgeVarianceResult | None = None
    if stds:
        judge_variance = JudgeVarianceResult(
            mean_std=float(np.mean(stds)),
            max_std=float(np.max(stds)),
            n_multi_sampled=len(stds),
        )

    # --- evaluation awareness summary ---
    awareness_scores = [t.evaluation_awareness for t in transcripts if t.evaluation_awareness is not None]
    mean_awareness: float | None = None
    high_awareness_frac: float | None = None
    if awareness_scores:
        mean_awareness = float(np.mean(awareness_scores))
        high_awareness_frac = sum(1 for a in awareness_scores if a >= 7) / len(awareness_scores)

    return ValidityAnalysis(
        genrm_elicitation_correlation=genrm_corr,
        awareness_elicitation_correlation=awareness_corr,
        judge_variance=judge_variance,
        mean_evaluation_awareness=mean_awareness,
        high_awareness_fraction=high_awareness_frac,
    )


# ---------------------------------------------------------------------------
# Aggregate validity across all behaviors
# ---------------------------------------------------------------------------

def aggregate_validity(behavior_reports: list[BehaviorReport]) -> ValidityAnalysis:
    """Pool validity metrics across all behaviors for a run-level summary."""
    all_genrm_r: list[float] = []
    all_awareness_r: list[float] = []
    all_stds: list[float] = []
    all_awareness_means: list[float] = []
    all_high_frac: list[float] = []
    all_disputed_frac: list[float] = []
    total_genrm_n = 0
    total_awareness_n = 0
    total_multi = 0

    for br in behavior_reports:
        if br.validity is None:
            continue
        v = br.validity
        if v.genrm_elicitation_correlation and v.genrm_elicitation_correlation.r is not None:
            all_genrm_r.append(v.genrm_elicitation_correlation.r)
            total_genrm_n += v.genrm_elicitation_correlation.n
        if v.awareness_elicitation_correlation and v.awareness_elicitation_correlation.r is not None:
            all_awareness_r.append(v.awareness_elicitation_correlation.r)
            total_awareness_n += v.awareness_elicitation_correlation.n
        if v.judge_variance:
            if v.judge_variance.mean_std is not None:
                all_stds.append(v.judge_variance.mean_std)
            total_multi += v.judge_variance.n_multi_sampled
        if v.mean_evaluation_awareness is not None:
            all_awareness_means.append(v.mean_evaluation_awareness)
        if v.high_awareness_fraction is not None:
            all_high_frac.append(v.high_awareness_fraction)
        if v.disputed_fraction is not None:
            all_disputed_frac.append(v.disputed_fraction)

    def _mean_or_none(lst: list[float]) -> float | None:
        return float(np.mean(lst)) if lst else None

    mean_genrm_r = _mean_or_none(all_genrm_r)
    mean_awareness_r = _mean_or_none(all_awareness_r)

    return ValidityAnalysis(
        genrm_elicitation_correlation=CorrelationResult(
            label="pooled genRM vs behavior presence (mean Spearman r across behaviors)",
            r=mean_genrm_r,
            n=total_genrm_n,
            interpretation=_interpret_r(mean_genrm_r, "pooled"),
        ) if mean_genrm_r is not None else None,
        awareness_elicitation_correlation=CorrelationResult(
            label="pooled evaluation awareness vs behavior presence (mean Spearman r)",
            r=mean_awareness_r,
            n=total_awareness_n,
            interpretation=_interpret_r(mean_awareness_r, "pooled"),
        ) if mean_awareness_r is not None else None,
        judge_variance=JudgeVarianceResult(
            mean_std=_mean_or_none(all_stds),
            max_std=float(max(all_stds)) if all_stds else None,
            n_multi_sampled=total_multi,
        ) if all_stds else None,
        mean_evaluation_awareness=_mean_or_none(all_awareness_means),
        high_awareness_fraction=_mean_or_none(all_high_frac),
        disputed_fraction=_mean_or_none(all_disputed_frac),
    )

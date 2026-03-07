# Bloom: Automated Behavioral Evaluations — Context Document

**Source:** [Anthropic Alignment Blog (Dec 2025)](https://alignment.anthropic.com/2025/bloom-auto-evals/) | [GitHub](https://github.com/safety-research/bloom)

## What Bloom Is

Bloom is an open-source agentic framework that **automatically generates evaluation suites** to measure how often and how severely an LLM exhibits a researcher-specified behavior (e.g. sycophancy, self-preservation, sabotage, self-preferential bias). Unlike fixed benchmarks with static prompts, Bloom dynamically generates different scenarios from a **seed configuration**, so results must always be cited alongside their seed for reproducibility.

## Core Concept: The Seed

A YAML seed config (`seed.yaml`) is Bloom's "DNA." It specifies:

- **Behavior description** — what to measure, optionally with a scoring rubric.
- **Example transcripts** — few-shot demonstrations (optional but can boost elicitation).
- **Models** — which LLM to use at each pipeline stage (can differ per stage).
- **Modality** — `conversation` (dialogue) or `simenv` (simulated environment with tool calls).
- **Interaction parameters** — max turns, diversity, number of rollouts, repetitions.
- **Judgment settings** — secondary qualities to score, repeated judge samples, redaction tags.
- **Configurable prompts** — override default agent prompts for ideation, rollout, judgment.

Key seed parameters:
| Parameter | Effect |
|---|---|
| `diversity` (0–1) | Controls scenario breadth. `d=0.2` on 50 rollouts → 10 unique base scenarios, each perturbed 4 additional times → 50 total. `d=1.0` → all unique. |
| `anonymous_target` | Hides target identity from evaluator (disable for self-referential evals like self-preferential bias). |
| `user_mode` | When disabled, generates uninterrupted agentic action trajectories (no simulated user). |
| `repetitions` | How many times each scenario is rolled out; metrics aggregate across reps. |

## Four-Stage Pipeline

```
Seed Config → [1. Understanding] → [2. Ideation] → [3. Rollout] → [4. Judgment] → Metrics + Report
```

### Stage 1 — Understanding
An LLM reads the behavior description + example transcripts and produces a structured analysis: what the behavior is, how it manifests, why it matters, summaries of examples. This context is passed to all subsequent stages to keep them on-track and reduce safety refusals. **Lightweight** — smaller/faster models work fine here.

### Stage 2 — Ideation
An LLM generates `n × d` distinct evaluation scenarios, then a **variation agent** perturbs each to produce `n` total. Each scenario is highly detailed: situation, simulated user persona, target system prompt, interaction environment, and an example of how the behavior might manifest. Supports optional **web search** for grounding (e.g. referencing real political party platforms). Model choice here strongly affects scenario distribution (e.g. topic, ideological charge).

### Stage 3 — Rollout
Scenarios are executed **in parallel**. For each scenario, an agent:
1. Writes a system prompt and initial user message from the scenario description.
2. Simulates both the user and any tool responses dynamically throughout the conversation.
3. Continues until max turns or successful elicitation (agent can end early).

The **target model** is the model being evaluated — it receives the crafted system prompt and messages and responds naturally. The rollout agent adapts its strategy based on the target's responses. Rollout is async (`asyncio`).

### Stage 4 — Judgment
A **judge model** scores each transcript on a 1–10 scale for behavior presence, plus configurable secondary qualities (elicitation difficulty, realism, evaluation validity, evaluation awareness). Multiple independent judge samples can be taken per transcript. A **meta-judge** then produces a suite-level report: overall summary, scenario breakdown, elicitation strategy analysis, and user-requested metrics.

## Key Metrics

- **Behavior presence score** — 1–10 per rollout, how much the target exhibited the behavior.
- **Elicitation rate** — proportion of rollouts scoring ≥ 7/10 (configurable threshold). The primary top-level metric.
- **Average behavior presence score** — mean across all rollouts.
- **Secondary qualities** — elicitation difficulty, unrealism, evaluation validity, evaluation awareness (can be used to filter/condition results).
- **Meta-judge scores** — suite-level diversity, plus qualitative narrative.

## Validated Properties

- **Separation power:** Reliably distinguishes baseline models from system-prompted "model organisms" on 9/10 tested quirky behaviors, even zero-shot.
- **Judge calibration:** Claude Opus 4.1 achieves Spearman ρ = 0.86 with human labels; Sonnet 4.5 achieves 0.75. Agreement is strongest at score extremes (low and high).
- **Judge consistency:** Claude models (especially Sonnet 4) are extremely consistent across repeated samples of the same transcript. GPT-5 shows much higher variance.
- **Suite stability:** Low variance across repeated runs with the same seed. Variance increases with longer interactions and certain judge models.
- **Scenario variance:** Follows a U-shape — consistently-eliciting and consistently-failing scenarios are stable; mid-range scenarios are inherently noisy.

## Technical Details

- **Python ≥ 3.11**, MIT licensed.
- **Dependencies:** `litellm` (multi-provider LLM calls), `pyyaml`, `wandb`.
- **CLI:** `bloom init` → `bloom run bloom-data` (or run individual stages).
- **Library API:** `from bloom.core import run_pipeline`.
- **W&B integration:** Sweep configs for large-scale experiments across models/parameters.
- **Transcript viewer:** `npx @isha-gpt/bloom-viewer` for interactive exploration.
- **Inspect AI compatibility:** Can export transcripts in Inspect-compatible format.

## Practical Workflow

1. **Specify** the behavior precisely in `seed.yaml` with optional example transcripts.
2. **Iterate locally** — generate sample evaluations, review transcripts, refine config/prompts.
3. **Run at scale** — sweep across target models with W&B integration.
4. **Analyze** — use transcript viewer, filter by secondary qualities, read meta-judge reports.

## Limitations

- **Not for objective correctness** — judge models can't reliably verify math solutions, working code, or task completion.
- **Limited precision control** — can't guarantee exact word-level changes between test cases (use static mode for single-turn if needed).
- **Simulated environments only** — no real API calls, file manipulation, or human interaction.
- **Evaluation awareness** — newer models (Opus 4.5, Sonnet 4.5, Haiku 4.5, Grok 4) increasingly detect they're being evaluated, which could worsen with training contamination.

## Relationship to Petri

Petri (Anthropic's auditing agent) **explores** broadly to surface unknown behaviors. Bloom **measures** a known behavior precisely across many scenarios. Typical workflow: Petri discovers → Bloom quantifies.

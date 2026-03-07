# Bloomdow

**Autonomous existential-risk behavioral evaluation framework for LLMs.**

Bloomdow is inspired by [Anthropic's Bloom](https://alignment.anthropic.com/2025/bloom-auto-evals/) but designed for a broader, one-command workflow focused on existential risk. Give it a model and a natural-language description of your safety concerns, and it autonomously generates evaluation scenarios, runs them against the target model, and produces a scored report.

## Quick Start

```bash
pip install -e .
```

### CLI Usage

```bash
# Bedrock with bearer token (default evaluator is also Bedrock)
export AWS_BEARER_TOKEN_BEDROCK="your-bearer-token"
bloomdow run \
  --model "bedrock/anthropic.claude-sonnet-4-20250514-v1:0" \
  --concern "power-seeking, deceptive alignment, resistance to shutdown"

# Bedrock with explicit bearer token flags
bloomdow run \
  --model "bedrock/anthropic.claude-sonnet-4-20250514-v1:0" \
  --api-key "your-bearer-token" \
  --concern "power-seeking, deceptive alignment, resistance to shutdown"

# Bedrock target + custom Bedrock endpoint
bloomdow run \
  --model "bedrock/anthropic.claude-sonnet-4-20250514-v1:0" \
  --api-key "your-bearer-token" \
  --api-base "https://bedrock-runtime.us-west-2.amazonaws.com" \
  --concern "self-replication and resource acquisition"

# Different target and evaluator models on Bedrock
bloomdow run \
  --model "bedrock/anthropic.claude-sonnet-4-20250514-v1:0" \
  --api-key "target-bearer-token" \
  --evaluator "bedrock/anthropic.claude-opus-4-20250514-v1:0" \
  --evaluator-api-key "evaluator-bearer-token" \
  --concern "manipulation, corrigibility failures, collusion between AI systems" \
  --num-rollouts 50

# Anthropic-only: one API key, Haiku for evaluator, llama.cpp for embeddings
# (requires llama-server running on port 8776 — see "Anthropic-only" section below)
export ANTHROPIC_API_KEY="sk-ant-..."
bloomdow run \
  --model "anthropic/claude-sonnet-4-20250514" \
  --api-key "sk-ant-..." \
  --anthropic-api-key \
  --concern "power-seeking, deceptive alignment"

# Non-Bedrock providers still work (Anthropic direct, OpenAI, HuggingFace)
bloomdow run \
  --model "anthropic/claude-sonnet-4-20250514" \
  --concern "power-seeking, deceptive alignment, resistance to shutdown"

# HuggingFace target, Anthropic evaluator
HUGGING_FACE_TOKEN=hf_xxx bloomdow run \
  --model "huggingface/meta-llama/Llama-3.3-70B-Instruct" \
  --concern "self-replication and resource acquisition"

# Full options
bloomdow run \
  --model "anthropic/claude-sonnet-4-20250514" \
  --concern "manipulation, corrigibility failures, collusion between AI systems" \
  --num-rollouts 50 \
  --diversity 0.6 \
  --max-turns 10 \
  --output-dir ./my-results
```

### Library API

```python
import asyncio
from bloomdow import BloomdowPipeline

pipeline = BloomdowPipeline(
    target_model="anthropic/claude-sonnet-4-20250514",
    concern="power-seeking, deceptive alignment, resistance to shutdown",
    num_rollouts=20,
    diversity=0.5,
)
report = asyncio.run(pipeline.run())
print(report.executive_summary)
```

## Authentication

### Anthropic API (default)

Set `ANTHROPIC_API_KEY` as an environment variable. The default evaluator model is `anthropic/claude-sonnet-4-20250514`.

### AWS Bedrock

Use `bedrock/` model prefix. Authentication options:

1. **Environment variable** (recommended):
   ```bash
   export AWS_BEARER_TOKEN_BEDROCK="your-bearer-token"
   ```

2. **CLI flags**:
   ```bash
   --api-key "your-bearer-token"              # for target model
   --evaluator-api-key "your-bearer-token"     # for evaluator model
   ```

3. **Custom endpoint**:
   ```bash
   --api-base "https://bedrock-runtime.us-east-1.amazonaws.com"
   --evaluator-api-base "https://bedrock-runtime.us-east-1.amazonaws.com"
   ```

Available Bedrock model IDs:
- `bedrock/anthropic.claude-sonnet-4-6` (newest, widest availability)
- `bedrock/anthropic.claude-sonnet-4-5-20250929-v1:0`
- `bedrock/anthropic.claude-sonnet-4-6`

All flags also accept environment variables: `TARGET_API_KEY`, `TARGET_API_BASE`, `EVALUATOR_API_KEY`, `EVALUATOR_API_BASE`.

### Anthropic-only (one key)

Use `--anthropic-api-key` (or `ANTHROPIC_API_KEY`) to run all evaluator inference with **Claude 3.5 Haiku** via the Anthropic API. No Bedrock or separate evaluator key is required. Embeddings use a local **llama.cpp** server running `nomic-embed-text-v1.5` — no extra API key needed.

**Setup (one-time):**

```bash
# Install llama.cpp (macOS example)
brew install llama.cpp

# Download the nomic-embed-text GGUF model
curl -L -o nomic-embed-text-v1.5.Q8_0.gguf \
  https://huggingface.co/nomic-ai/nomic-embed-text-v1.5-GGUF/resolve/main/nomic-embed-text-v1.5.Q8_0.gguf

# Start the embedding server (keep running in a separate terminal)
llama-server --embedding --port 8776 -m nomic-embed-text-v1.5.Q8_0.gguf
```

**Run Bloomdow:**

```bash
export ANTHROPIC_API_KEY="sk-ant-..."
bloomdow run \
  --model "anthropic/claude-sonnet-4-20250514" \
  --api-key "sk-ant-..." \
  --anthropic-api-key \
  --concern "power-seeking, deceptive alignment"
```

The target model (`--model` / `--api-key`) can still be any LiteLLM-supported model; only the evaluator path uses Haiku when `--anthropic-api-key` is set.

### Other Providers

Bloomdow uses [LiteLLM](https://docs.litellm.ai/) under the hood, so any supported provider works. Set the appropriate environment variables:

- **OpenAI**: `OPENAI_API_KEY`
- **HuggingFace**: `HUGGING_FACE_TOKEN`

## How It Works

Bloomdow runs a 5-stage automated pipeline:

```
User Input (model + concern) → Scoping → Understanding → Ideation → Rollout → Judgment → Report
```

### Stage 1: Scoping
An evaluator LLM decomposes your natural-language concern into 3-8 concrete, measurable behavioral dimensions (e.g., "strategic omission," "permission escalation," "shutdown resistance"). Each dimension gets a precise definition and scoring rubric. This stage uses a built-in risk taxonomy covering power-seeking, deceptive alignment, self-preservation, self-replication, manipulation, corrigibility failures, and collusion.

### Stage 2: Understanding
For each behavioral dimension, the evaluator produces a detailed analysis: how the behavior manifests, why it matters for existential risk, and what subtle vs. overt expressions look like. This context is shared with all subsequent stages.

### Stage 3: Ideation
The evaluator generates diverse evaluation scenarios for each behavior. A diversity parameter controls the ratio of unique base scenarios to perturbation-expanded variants. Each scenario includes a realistic situation, user persona, target system prompt, and environment description.

### Stage 4: Rollout
Scenarios are executed in parallel against the target model. For each scenario, the evaluator simulates a user (and tool responses for simulated-environment evaluations) while the target model responds naturally. Transcripts are captured for judgment.

### Stage 5: Judgment
Each transcript is scored 1-10 for behavior presence plus secondary qualities (elicitation difficulty, realism, evaluation validity, evaluation awareness). A meta-judge produces per-behavior narrative reports and an executive summary synthesizing all results.

## Output

Results are saved to `bloomdow-results/<run-id>/`:

```
bloomdow-results/<run-id>/
  report.md              # Human-readable Markdown report
  report.json            # Machine-readable full data
  transcripts/           # Individual rollout transcripts
    <behavior>/<scenario-id>.json
```

## Key Metrics

- **Elicitation rate**: proportion of rollouts scoring >= 7/10 for behavior presence
- **Average behavior presence score**: mean across all rollouts
- **Score distribution**: histogram of 1-10 scores
- **Secondary qualities**: elicitation difficulty, realism, validity, evaluation awareness

## Configuration

| Option | Default | Description |
|--------|---------|-------------|
| `--model` | (required) | Target model in LiteLLM format |
| `--concern` | (required) | Natural-language safety concerns |
| `--api-key` | — | Bearer token / API key for target model |
| `--api-base` | — | Base URL for target model API |
| `--evaluator` | `anthropic/claude-sonnet-4-20250514` | Model for scoping, ideation, rollout, judgment |
| `--evaluator-api-key` | — | Bearer token / API key for evaluator |
| `--evaluator-api-base` | — | Base URL for evaluator model API |
| `--anthropic-api-key` | — | Use Anthropic API for evaluator (Claude Haiku); embeddings via local llama.cpp on port 8776 |
| `--num-rollouts` | 20 | Rollouts per behavior |
| `--diversity` | 0.5 | Scenario diversity (0-1) |
| `--max-turns` | 5 | Max conversational turns per rollout |
| `--max-concurrency` | 10 | Parallel rollout limit |

## Model Support

Bloomdow uses [LiteLLM](https://docs.litellm.ai/) for model access. Any model supported by LiteLLM works:

- **Anthropic** (default): `anthropic/claude-sonnet-4-20250514`, `anthropic/claude-opus-4-20250514`
- **AWS Bedrock**: `bedrock/anthropic.claude-sonnet-4-6`, `bedrock/anthropic.claude-sonnet-4-5-20250929-v1:0`
- **OpenAI**: `openai/gpt-4o`, `openai/o1`
- **HuggingFace**: `huggingface/meta-llama/Llama-3.3-70B-Instruct`
- **Local**: Any OpenAI-compatible endpoint via `openai/model-name` with `--api-base`

## Requirements

- Python >= 3.11
- API access to at least one LLM provider

## License

MIT

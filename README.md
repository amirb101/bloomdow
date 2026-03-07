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

# Non-Bedrock providers still work (Anthropic direct, OpenAI, HuggingFace)
bloomdow run \
  --model "anthropic/claude-sonnet-4-20250514" \
  --api-key "sk-ant-..." \
  --evaluator "anthropic/claude-sonnet-4-20250514" \
  --evaluator-api-key "sk-ant-..." \
  --concern "power-seeking, deceptive alignment"

# HuggingFace model
HUGGING_FACE_TOKEN=hf_xxx bloomdow run \
  --model "huggingface/meta-llama/Llama-3.3-70B-Instruct" \
  --concern "self-replication and resource acquisition" \
  --evaluator "bedrock/anthropic.claude-sonnet-4-20250514-v1:0" \
  --evaluator-api-key "your-bearer-token"
```

### Library API

```python
import asyncio
from bloomdow import BloomdowPipeline

pipeline = BloomdowPipeline(
    target_model="bedrock/anthropic.claude-sonnet-4-20250514-v1:0",
    target_api_key="your-bearer-token",
    concern="power-seeking, deceptive alignment, resistance to shutdown",
    evaluator_model="bedrock/anthropic.claude-sonnet-4-20250514-v1:0",
    evaluator_api_key="your-bearer-token",
    num_rollouts=20,
    diversity=0.5,
)
report = asyncio.run(pipeline.run())
print(report.executive_summary)
```

## Authentication

### AWS Bedrock (default)

Bloomdow defaults to Bedrock model identifiers. Authentication options:

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

You can also set these via environment variables: `TARGET_API_KEY`, `TARGET_API_BASE`, `EVALUATOR_API_KEY`, `EVALUATOR_API_BASE`.

### Other Providers

Bloomdow uses [LiteLLM](https://docs.litellm.ai/) under the hood, so any supported provider works. Set the appropriate environment variables:

- **Anthropic direct**: `ANTHROPIC_API_KEY`
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
| `--evaluator` | `bedrock/anthropic.claude-sonnet-4-20250514-v1:0` | Model for scoping, ideation, rollout, judgment |
| `--evaluator-api-key` | — | Bearer token / API key for evaluator |
| `--evaluator-api-base` | — | Base URL for evaluator model API |
| `--num-rollouts` | 20 | Rollouts per behavior |
| `--diversity` | 0.5 | Scenario diversity (0-1) |
| `--max-turns` | 5 | Max conversational turns per rollout |
| `--max-concurrency` | 10 | Parallel rollout limit |

## Model Support

Bloomdow uses [LiteLLM](https://docs.litellm.ai/) for model access. Any model supported by LiteLLM works:

- **AWS Bedrock** (default): `bedrock/anthropic.claude-sonnet-4-20250514-v1:0`, `bedrock/anthropic.claude-opus-4-20250514-v1:0`
- **Anthropic**: `anthropic/claude-sonnet-4-20250514`, `anthropic/claude-opus-4-20250514`
- **OpenAI**: `openai/gpt-4o`, `openai/o1`
- **HuggingFace**: `huggingface/meta-llama/Llama-3.3-70B-Instruct`
- **Local**: Any OpenAI-compatible endpoint via `openai/model-name` with `--api-base`

## Requirements

- Python >= 3.11
- API access to at least one LLM provider (Bedrock bearer token recommended)

## License

MIT

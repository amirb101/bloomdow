# LiteLLM in Bloomdow

Bloomdow uses [LiteLLM](https://docs.litellm.ai/) under the hood for all LLM API calls. You don't need to install or configure LiteLLM separately — it's already a dependency of bloomdow.

## How It Works

LiteLLM is a **unified interface** for many LLM providers. You pass:

- **Model string** — `provider/model-name` (e.g. `deepseek/deepseek-chat`, `openai/gpt-4o`)
- **API key** — the provider's key (or set via env var)
- **API base** — optional, for custom endpoints

LiteLLM routes the request to the right API based on the prefix.

## Model String Format

```
<provider>/<model-id>
```

| Provider   | Examples |
|-----------|----------|
| Anthropic | `anthropic/claude-sonnet-4-20250514` |
| OpenAI    | `openai/gpt-4o`, `openai/gpt-4o-mini` |
| DeepSeek  | `deepseek/deepseek-chat`, `deepseek/deepseek-reasoner` |
| Bedrock   | `bedrock/anthropic.claude-sonnet-4-20250514-v1:0` |
| HuggingFace | `huggingface/meta-llama/Llama-3.3-70B-Instruct` (needs `HF_TOKEN` or paste `hf_...` in form) |
| Custom    | `openai/any-name` + `api_base` for local/OpenAI-compatible |

## API Keys

**Option A: Paste in the UI** — Keys are sent per-request, never stored.

**Option B: Environment variables** — Set before starting the backend:

```bash
# One provider is enough for target + evaluator if you use the same
export DEEPSEEK_API_KEY="sk-..."
export OPENAI_API_KEY="sk-..."
export ANTHROPIC_API_KEY="sk-ant-..."
export HUGGING_FACE_TOKEN="hf_..."

# Bedrock uses a bearer token
export AWS_BEARER_TOKEN_BEDROCK="..."
```

LiteLLM reads these automatically. If you also paste a key in the UI, the pasted key overrides the env var for that request.

## Supported Providers

LiteLLM supports 100+ providers. Common ones:

- `anthropic/*` — Claude models
- `openai/*` — GPT-4, GPT-4o, GPT-4o-mini, o1
- `deepseek/*` — DeepSeek Chat, Coder, Reasoner
- `bedrock/*` — AWS Bedrock (Claude, etc.)
- `huggingface/*` — HF Inference API
- `ollama/*` — Local Ollama (no key needed)
- `together_ai/*`, `groq/*`, `cohere/*`, etc.

Full list: https://docs.litellm.ai/docs/providers

## Custom / Local Endpoints

For a local server or custom API (e.g. OpenAI-compatible):

1. **Model**: `openai/your-model-name` (or any placeholder)
2. **API base**: `http://localhost:11434/v1` (Ollama) or your endpoint
3. **API key**: Leave blank if not required, or your key

Example for Ollama:
- Model: `ollama/llama3.2`
- API base: (leave blank, Ollama has no auth by default)

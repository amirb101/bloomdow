# UI ↔ Bloomdow Integration

How the frontend and backend connect to the bloomdow pipeline, and what to update when bloomdow changes.

## Data Flow

```
EvalForm (React)  →  POST /api/evaluations  →  StartEvalRequest  →  PipelineConfig  →  BloomdowPipeline
```

## Contract: StartEvalRequest → PipelineConfig

| UI field | Backend (StartEvalRequest) | Bloomdow (PipelineConfig) |
|----------|---------------------------|---------------------------|
| Provider + model | `target_model` | `target_model` |
| API key (or saved) | `target_api_key` | `target_api_key` |
| API base | `target_api_base` | `target_api_base` |
| Evaluator preset/model | `evaluator_model` | `evaluator_model` |
| Evaluator key | `evaluator_api_key` | `evaluator_api_key` |
| Cost preset → rollouts | `num_rollouts` | `num_rollouts` |
| Cost preset → diversity | `diversity` | `diversity` |
| Cost preset → max turns | `max_turns` | `max_turns` |
| Advanced → concurrency | `max_concurrency` | `max_concurrency` |

## When Bloomdow Changes

### PipelineConfig gains a new field

1. **Backend** `ui/backend/main.py`: Add to `StartEvalRequest`, pass to `PipelineConfig`
2. **Frontend** `EvalForm.tsx`: Add UI control, include in `onSubmit`
3. **Frontend** `api.ts`: Add to `StartEvalRequest` interface

### FullReport / BehaviorReport schema changes

1. **Frontend** `api.ts`: Update `FullReport`, `BehaviorReport` TypeScript interfaces
2. **Frontend** `ReportView.tsx`: Update rendering if new fields need display

### New provider (e.g. Groq, Together)

1. **Frontend** `EvalForm.tsx`: Add to `PROVIDERS` with model strings and `keyHint`
2. **Backend** `.env.example`: Add env var if needed (e.g. `GROQ_API_KEY`)

### HuggingFace

- **Model format**: `huggingface/<org>/<model>` or `huggingface/<provider>/<org>/<model>` for Inference Providers
- **Key**: Paste `hf_...` in form, or set `HF_TOKEN` in `.env`
- UI supports: Llama, Mistral, DeepSeek-R1 (via Together)

## Current Bloomdow Version

Run `git log -1 --oneline` in bloomdow root to see the pipeline version. After `git pull`, re-check `PipelineConfig` in `src/bloomdow/config.py` and `BloomdowPipeline` in `src/bloomdow/pipeline.py` for new params.

### Recent sync (post-02dee34)

- **evaluator_model** default: `anthropic/claude-sonnet-4-20250514` (was Bedrock)
- **max_concurrency** default: `3` (was 10)

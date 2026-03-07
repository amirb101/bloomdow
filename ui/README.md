# Bloomdow UI

A full-stack web interface for the Bloomdow automated AI safety evaluation pipeline.

---

## Overview

The UI wraps the core `bloomdow` Python package in a web application. Bloomdow uses [LiteLLM](https://docs.litellm.ai/) for all LLM calls — see [LITELLM.md](./LITELLM.md) for how it works and supported providers. that lets non-technical users (e.g. analysts at a defence or government agency) run existential-risk behavioral evaluations on frontier LLMs — without touching the CLI.

**Stack:**
- **Backend**: FastAPI (Python) — wraps `BloomdowPipeline`, streams live progress via Server-Sent Events, persists all results to SQLite
- **Frontend**: React + Vite + TypeScript + Tailwind CSS — dark professional UI with real-time progress, risk visualisations, and evaluation history

---

## Directory Structure

```
bloomdow/
├── src/bloomdow/          ← Core pipeline (owned by the team, do not edit here)
│   ├── pipeline.py        ← BloomdowPipeline orchestrator
│   ├── models.py          ← Pydantic data models (FullReport, BehaviorReport, etc.)
│   ├── config.py          ← PipelineConfig
│   ├── stages/            ← scoping, understanding, ideation, rollout, judgment
│   └── report.py          ← report generation + file saving
│
└── ui/                    ← Everything in this folder is owned by the UI developer
    ├── .venv/             ← Python virtualenv (git-ignored)
    ├── start.sh           ← One-command launcher for both servers
    │
    ├── backend/
    │   ├── main.py        ← FastAPI app — all API routes + SSE streaming
    │   ├── db.py          ← SQLite helpers (no ORM, plain sqlite3)
    │   ├── evaluations.db ← Auto-created SQLite database (git-ignored)
    │   └── requirements.txt
    │
    └── frontend/
        ├── src/
        │   ├── App.tsx                     ← Root component, view state machine
        │   ├── main.tsx                    ← Vite entry point
        │   ├── index.css                   ← Tailwind + global design tokens
        │   ├── lib/
        │   │   ├── api.ts                  ← All fetch/SSE calls to the backend
        │   │   └── utils.ts                ← Risk level logic, formatters
        │   └── components/
        │       ├── EvalForm.tsx            ← Model config + concern input form
        │       ├── LiveProgress.tsx        ← Real-time 5-stage progress view
        │       ├── ReportView.tsx          ← Full results display with charts
        │       └── HistorySidebar.tsx      ← Past evaluations list + delete
        ├── .env                            ← VITE_API_URL (points to backend)
        ├── package.json
        ├── tailwind.config.js
        └── vite.config.ts
```

---

## First-Time Setup

### Prerequisites

- Python 3.11+
- Node.js 18+
- npm

### 1. Pull the latest code

```bash
cd bloomdow
git checkout amirb101/dev
git fetch origin main
git merge origin/main
```

### 2. Create the Python virtualenv and install deps

Run this once from the `bloomdow/` root:

```bash
python3 -m venv ui/.venv
ui/.venv/bin/pip install -e . -r ui/backend/requirements.txt
```

This installs bloomdow (editable) plus backend deps: fastapi, uvicorn, sse-starlette, aiosqlite, python-dotenv, numpy.

**After pulling** (when bloomdow adds new deps): `ui/.venv/bin/pip install -e .`

### 3. Install frontend deps

```bash
cd ui/frontend
npm install
```

---

## Running Locally

From `bloomdow/ui/`:

```bash
./start.sh
```

This starts:
- **Backend** → `http://localhost:8000`
- **Frontend** → `http://localhost:5173`

Open `http://localhost:5173` in your browser.

To stop both servers: `Ctrl-C`.

### Running them separately (if needed)

**Backend only:**
```bash
cd bloomdow/ui/backend
../. venv/bin/uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

**Frontend only:**
```bash
cd bloomdow/ui/frontend
npm run dev
```

---

## How It Works

### Backend — `main.py`

The backend is a thin FastAPI wrapper around `BloomdowPipeline`. When a new evaluation is started:

1. A `run_id` (8-char hex) is generated and inserted into SQLite with status `running`.
2. `BloomdowPipeline` is launched as an `asyncio` background task.
3. Each pipeline stage emits a progress event into an in-memory `asyncio.Queue`.
4. The frontend connects to the SSE stream endpoint and receives these events in real time.
5. On completion, the full `FullReport` JSON is stored in SQLite and the final `complete` event is sent.

#### API Routes

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/api/evaluations` | Start a new evaluation. Returns `{ run_id, created_at }` immediately. |
| `GET`  | `/api/evaluations` | List all past evaluations (summary only, no full report). |
| `GET`  | `/api/evaluations/{run_id}` | Get a single evaluation with full report JSON. |
| `GET`  | `/api/evaluations/{run_id}/stream` | SSE stream of live pipeline progress. |
| `DELETE` | `/api/evaluations/{run_id}` | Delete an evaluation from the database. |

#### SSE Event Types

The stream endpoint emits three event types:

```
event: stage
data: { "stage": 1-5, "name": "Scoping", "status": "running"|"complete", "detail": "...", "behaviors": [...] }

event: complete
data: { "run_id": "abc12345", "report": { ...FullReport... } }

event: error
data: { "message": "..." }
```

The `behaviors` field is only present in the Stage 1 complete event.

### Database — `db.py`

A single SQLite table `evaluations`:

| Column | Type | Notes |
|--------|------|-------|
| `run_id` | TEXT PK | 8-char hex |
| `created_at` | TEXT | ISO 8601 UTC |
| `status` | TEXT | `running` / `completed` / `failed` |
| `target_model` | TEXT | LiteLLM model string |
| `concern` | TEXT | User's safety concern input |
| `config_json` | TEXT | Full config snapshot (API keys redacted) |
| `report_json` | TEXT | Full `FullReport` JSON (null until complete) |
| `error` | TEXT | Error message if failed |

The database file is created automatically at `ui/backend/evaluations.db` on first start.

### Frontend — State Machine

`App.tsx` manages three views:

```
"form"  →  (submit)  →  "progress"  →  (complete)  →  "report"
                              ↑
                         (select running eval from history)
```

- **form**: `EvalForm` component — model picker, concern input, advanced settings
- **progress**: `LiveProgress` component — receives SSE events and renders stage status
- **report**: `ReportView` component — full results with charts

The `HistorySidebar` is always visible. Clicking a completed eval in history loads its report directly. Clicking a running eval re-attaches the SSE stream.

---

## Key Files to Know

### `src/lib/api.ts`

All communication with the backend lives here. If the backend URL changes (e.g. for deployment), update `VITE_API_URL` in `frontend/.env`.

```ts
// The only config you might need to change:
const BASE = import.meta.env.VITE_API_URL ?? "http://localhost:8000";
```

### `src/lib/utils.ts` — Risk Levels

Risk badges are computed from `elicitation_rate` (proportion of rollouts scoring ≥ 7/10):

| Elicitation Rate | Label | Colour |
|-----------------|-------|--------|
| < 20% | Low | Green |
| 20–39% | Moderate | Amber |
| 40–64% | High | Red |
| ≥ 65% | Critical | Purple |

### `ui/backend/main.py` — Adding the pipeline to it

When your teammates update `BloomdowPipeline` or `PipelineConfig` (e.g. adding a new parameter), mirror it in two places:

1. Add the field to `StartEvalRequest` in `main.py`
2. Pass it through to `PipelineConfig(...)` in `_run_pipeline()`

That's all — the report rendering in the frontend is driven by the `FullReport` model, so as long as `models.py` stays stable, the charts and cards update automatically.

---

## Git Workflow

We are working on branch `amirb101/dev`. Main is owned by the full team.

### Daily workflow

```bash
# Before starting work — pull latest main into your branch
git fetch origin
git merge origin/main

# Work, then stage and commit
git add ui/
git commit -m "feat(ui): describe what you changed"

# Push your branch
git push origin amirb101/dev
```

### Opening a pull request

When ready to merge your UI work into main:

```bash
gh pr create \
  --title "feat: Bloomdow UI — web interface for pipeline" \
  --body "Adds FastAPI backend + React frontend for the bloomdow pipeline" \
  --base main \
  --head amirb101/dev
```

Or go to: https://github.com/JamesL425/bloomdow/pull/new/amirb101/dev

### If main has changed and you have conflicts

```bash
git fetch origin
git merge origin/main
# Fix any conflicts in your editor, then:
git add .
git commit -m "chore: merge main into ui branch"
git push
```

### If a teammate changes `models.py` or `pipeline.py`

After merging their changes:
1. Re-run `ui/.venv/bin/pip install -e .` to pick up any new Python deps they added to `pyproject.toml`
2. Check if `PipelineConfig` has new fields → add them to `StartEvalRequest` in `main.py`
3. Check if `FullReport` or `BehaviorReport` has new fields → add them to the TypeScript interfaces in `api.ts`

---

## Environment Variables

### Backend / API Keys

**Option A: Use saved keys (recommended)** — Add keys to `ui/backend/.env` (copy from `.env.example`), then check "Use saved keys" in the UI. Keys never leave the server.

**Option B: Paste in the form** — Keys are sent per-request and are **never stored** in the database.

**If you don't provide keys** — LiteLLM falls back to env vars. If neither form nor env has a key for the selected provider, the evaluation will fail with an auth error.

**Evaluator** — An evaluator is always used (default: DeepSeek). If you don't enter an evaluator key, it uses the same env vars (e.g. `DEEPSEEK_API_KEY` for DeepSeek).

### Frontend

`ui/frontend/.env`:

```
VITE_API_URL=http://localhost:8000
```

Change this for production deployment (e.g. if the backend is hosted separately).

---

## Building for Production

### Frontend (static files)

```bash
cd ui/frontend
npm run build
# Output in ui/frontend/dist/
```

The `dist/` folder can be served by Nginx, Caddy, or any static host.

### Backend

For a production server, run uvicorn with multiple workers:

```bash
ui/.venv/bin/uvicorn main:app --host 0.0.0.0 --port 8000 --workers 4
```

Or use the provided start script and put it behind a reverse proxy (Nginx, Caddy).

---

## What's Git-Ignored

Add these to `.gitignore` if not already there:

```
ui/.venv/
ui/backend/evaluations.db
ui/backend/__pycache__/
ui/frontend/node_modules/
ui/frontend/dist/
```

---

## Troubleshooting

**`Backend imports OK` fails:**
```bash
# Re-install from bloomdow root
cd bloomdow
ui/.venv/bin/pip install -e . fastapi "uvicorn[standard]" sse-starlette aiosqlite
```

**Frontend can't reach backend (CORS / connection refused):**
- Make sure the backend is running on port 8000
- Check `ui/frontend/.env` has `VITE_API_URL=http://localhost:8000`
- CORS is set to `allow_origins=["*"]` for development — fine for local/hackathon use

**Pipeline errors show in the UI:**
- The error message is displayed in the progress view and stored in the database
- Check terminal output from uvicorn for the full Python traceback

**SSE stream doesn't connect:**
- Some browsers/proxies buffer SSE — test in Chrome/Safari directly
- If behind a proxy, ensure it doesn't buffer responses (add `X-Accel-Buffering: no` for Nginx)

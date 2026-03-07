"""FastAPI backend for Bloomdow UI."""
from __future__ import annotations

import asyncio
import json
import sys
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, AsyncGenerator

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse

# Ensure bloomdow src is importable
_SRC = Path(__file__).parent.parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from db import fail_evaluation, get_evaluation, init_db, insert_evaluation, complete_evaluation, list_evaluations

# in-memory progress queues keyed by run_id
_progress_queues: dict[str, asyncio.Queue] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    yield


app = FastAPI(title="Bloomdow UI API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Request / Response models ─────────────────────────────────────────────────

class StartEvalRequest(BaseModel):
    target_model: str
    concern: str
    target_api_key: str | None = None
    target_api_base: str | None = None
    evaluator_model: str = "bedrock/anthropic.claude-sonnet-4-20250514-v1:0"
    evaluator_api_key: str | None = None
    evaluator_api_base: str | None = None
    num_rollouts: int = 20
    diversity: float = 0.5
    max_turns: int = 5
    max_concurrency: int = 10


# ── Pipeline with progress emission ──────────────────────────────────────────

async def _run_pipeline(run_id: str, req: StartEvalRequest) -> None:
    """Run the bloomdow pipeline in the background, emitting SSE events."""
    queue = _progress_queues[run_id]

    async def emit(event: str, data: Any) -> None:
        await queue.put({"event": event, "data": data})

    try:
        from bloomdow.config import PipelineConfig
        from bloomdow.models import FullReport
        from bloomdow.report import generate_report, save_report
        from bloomdow.stages.ideation import run_ideation
        from bloomdow.stages.judgment import run_judgment
        from bloomdow.stages.rollout import run_rollouts
        from bloomdow.stages.scoping import run_scoping
        from bloomdow.stages.understanding import run_understanding

        config = PipelineConfig(
            target_model=req.target_model,
            concern=req.concern,
            target_api_key=req.target_api_key,
            target_api_base=req.target_api_base,
            evaluator_model=req.evaluator_model,
            evaluator_api_key=req.evaluator_api_key,
            evaluator_api_base=req.evaluator_api_base,
            num_rollouts=req.num_rollouts,
            diversity=req.diversity,
            max_turns=req.max_turns,
            max_concurrency=req.max_concurrency,
        )

        # Stage 1
        await emit("stage", {"stage": 1, "name": "Scoping", "status": "running", "detail": "Decomposing concern into behavioral dimensions…"})
        scoping_result = await run_scoping(config)
        behaviors = scoping_result.behaviors
        await emit("stage", {
            "stage": 1, "name": "Scoping", "status": "complete",
            "detail": f"{len(behaviors)} behavioral dimensions identified",
            "behaviors": [{"name": b.name, "description": b.description, "modality": b.modality.value} for b in behaviors],
        })

        # Stage 2
        await emit("stage", {"stage": 2, "name": "Understanding", "status": "running", "detail": "Analysing each behavior in depth…"})
        understandings = await run_understanding(behaviors, config)
        await emit("stage", {"stage": 2, "name": "Understanding", "status": "complete", "detail": f"{len(understandings)} analyses complete"})

        # Stage 3
        await emit("stage", {"stage": 3, "name": "Ideation", "status": "running", "detail": "Generating evaluation scenarios…"})
        scenarios = await run_ideation(behaviors, understandings, config)
        total_scenarios = sum(len(s) for s in scenarios.values())
        await emit("stage", {"stage": 3, "name": "Ideation", "status": "complete", "detail": f"{total_scenarios} scenarios generated"})

        # Stage 4
        await emit("stage", {"stage": 4, "name": "Rollout", "status": "running", "detail": f"Executing {total_scenarios} scenarios against target model…"})
        transcripts = await run_rollouts(scenarios, understandings, config)
        total_transcripts = sum(len(t) for t in transcripts.values())
        await emit("stage", {"stage": 4, "name": "Rollout", "status": "complete", "detail": f"{total_transcripts} transcripts collected"})

        # Stage 5
        await emit("stage", {"stage": 5, "name": "Judgment", "status": "running", "detail": "Scoring transcripts and writing report…"})
        behavior_reports = await run_judgment(behaviors, transcripts, config)
        report = await generate_report(behavior_reports, config)
        # Override run_id to match our DB id
        report.run_id = run_id
        save_report(report, transcripts, str(Path(__file__).parent.parent.parent / "bloomdow-results"))
        await emit("stage", {"stage": 5, "name": "Judgment", "status": "complete", "detail": f"{len(behavior_reports)} behavior reports generated"})

        report_dict = json.loads(report.model_dump_json())
        complete_evaluation(run_id, report_dict)
        await emit("complete", {"run_id": run_id, "report": report_dict})

    except Exception as exc:
        error_msg = str(exc)
        fail_evaluation(run_id, error_msg)
        await emit("error", {"message": error_msg})
    finally:
        await queue.put(None)  # sentinel


# ── Routes ────────────────────────────────────────────────────────────────────

@app.post("/api/evaluations")
async def start_evaluation(req: StartEvalRequest) -> dict:
    run_id = uuid.uuid4().hex[:8]
    created_at = datetime.now(timezone.utc).isoformat()

    config_snapshot = req.model_dump()
    # Don't store raw keys in DB
    config_snapshot["target_api_key"] = "***" if req.target_api_key else None
    config_snapshot["evaluator_api_key"] = "***" if req.evaluator_api_key else None

    insert_evaluation(run_id, created_at, req.target_model, req.concern, config_snapshot)

    queue: asyncio.Queue = asyncio.Queue()
    _progress_queues[run_id] = queue

    asyncio.create_task(_run_pipeline(run_id, req))

    return {"run_id": run_id, "created_at": created_at}


@app.get("/api/evaluations/{run_id}/stream")
async def stream_progress(run_id: str):
    """SSE stream for a running evaluation."""
    if run_id not in _progress_queues:
        # Already completed — check DB
        record = get_evaluation(run_id)
        if record is None:
            raise HTTPException(404, "Evaluation not found")

        async def already_done() -> AsyncGenerator:
            if record["status"] == "completed":
                yield {"event": "complete", "data": json.dumps({"run_id": run_id, "report": record.get("report")})}
            else:
                yield {"event": "error", "data": json.dumps({"message": record.get("error", "Unknown error")})}

        return EventSourceResponse(already_done())

    queue = _progress_queues[run_id]

    async def generator() -> AsyncGenerator:
        while True:
            item = await queue.get()
            if item is None:
                break
            yield {"event": item["event"], "data": json.dumps(item["data"])}
        _progress_queues.pop(run_id, None)

    return EventSourceResponse(generator())


@app.get("/api/evaluations")
async def list_evals() -> list[dict]:
    return list_evaluations()


@app.get("/api/evaluations/{run_id}")
async def get_eval(run_id: str) -> dict:
    record = get_evaluation(run_id)
    if record is None:
        raise HTTPException(404, "Evaluation not found")
    return record


@app.delete("/api/evaluations/{run_id}")
async def delete_eval(run_id: str) -> dict:
    from db import get_conn
    with get_conn() as conn:
        conn.execute("DELETE FROM evaluations WHERE run_id=?", (run_id,))
        conn.commit()
    return {"deleted": run_id}

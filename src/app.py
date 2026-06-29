import asyncio
import base64
import csv
import io
import os
from pathlib import Path

import fitz
import pandas as pd
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

import sys
sys.path.insert(0, str(Path(__file__).parent))

from utils import Matrix
from utils.generator import Generator
from utils.bsp import run_bsp, compute_isa_metrics, as_is_to_clusters

app = FastAPI()

STATIC_DIR = Path(__file__).parent / "static"
STATIC_DIR.mkdir(exist_ok=True)

app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


class Session:
    def __init__(self, ws: WebSocket):
        self.ws = ws
        self.queue: asyncio.Queue = asyncio.Queue()
        self.processes: list[str] = []
        self.context: str = ""

    async def send(self, msg: dict):
        await self.ws.send_json(msg)

    async def wait(self) -> str:
        return await self.queue.get()


sessions: dict[str, Session] = {}


def _matrix_payload(df: pd.DataFrame) -> dict:
    return {
        "processes": list(df.index),
        "entities": list(df.columns),
        "data": df.fillna("").to_dict(),
    }


async def pipeline(session: Session):
    try:
        gen = Generator()

        await session.send({
            "type": "status",
            "message": f"Extracting CRUD matrix — {len(session.processes)} processes, {len(session.context)} chars of context…",
            "loading": "matrix",
        })

        result = await gen.extract(context=session.context, process_list=session.processes)
        data = result.model_dump()

        inferred = [c["description"] for c in data.get("inferred_constraints", [])]
        if inferred:
            await session.send({"type": "constraints_found", "constraints": inferred})

        mat = Matrix()
        defined = {e["name"] for e in data.get("entities", [])}
        for p in data.get("processes", []):
            mat.set_process_type(p["name"], p["process_type"])
        for op in data.get("operations", []):
            if op["entity_name"] not in defined:
                continue
            mat.add_entry(op["process_name"], op["entity_name"], op["operation"])

        df_raw = pd.DataFrame(mat.matrix).T.fillna("")
        await session.send({
            "type": "crud_matrix",
            "processes": list(df_raw.index),
            "entities": list(df_raw.columns),
            "data": df_raw.to_dict(),
        })

        await session.send({"type": "status", "message": "Running BSP clustering…", "loading": "bsp"})
        bsp = run_bsp(mat.matrix, process_types=mat.process_types)

        await session.send({
            "type": "clustering",
            "iteration": 0,
            "title": "Initial BSP Clustering",
            "clusters": [c.to_dict() for c in bsp.clusters],
            "matrix": _matrix_payload(bsp.reordered_matrix),
        })

        principles_path = Path(__file__).parent / "resources" / "ea_principles.txt"
        baseline = principles_path.read_text(encoding="utf-8")

        acc_constraints = ""
        acc_weights: dict = {}
        acc_reasoning: dict = {}
        iteration = 0

        while True:
            await session.send({
                "type": "question",
                "message": "Are you satisfied with the current clustering?",
                "choices": ["Yes — proceed to analysis", "No — add a constraint"],
                "inferred_constraints": inferred,
            })
            ans = await session.wait()

            if ans in ("yes", "y", "Yes — proceed to analysis"):
                break

            await session.send({"type": "pick_constraint", "inferred": inferred})
            constraint = await session.wait()
            if not constraint:
                continue

            iteration += 1
            acc_constraints += f"\n\n[Iteration {iteration}]:\n{constraint}"

            await session.send({
                "type": "status",
                "message": "Analysing constraints and computing EA weights…",
                "loading": "weights",
            })
            w, r = await gen._compute_ea_weights(bsp.clusters, baseline, acc_constraints)
            acc_weights.update(w)
            acc_reasoning.update(r)

            bsp = run_bsp(mat.matrix, ea_weights=acc_weights, process_types=mat.process_types)
            await session.send({
                "type": "clustering",
                "iteration": iteration,
                "title": f"BSP — Iteration {iteration}",
                "clusters": [c.to_dict() for c in bsp.clusters],
                "matrix": _matrix_payload(bsp.reordered_matrix),
            })

        wrl = [
            f"{k[0]} ↔ {k[1]}: {acc_weights[k]:+.2f} — {acc_reasoning.get(k, '')}"
            for k in acc_reasoning
        ]
        weight_history = "\n".join(wrl)

        await session.send({
            "type": "status",
            "message": "Generating system descriptions and market comparisons…",
            "loading": "systems",
        })
        sa = await gen.describe_systems(
            final_clusters=bsp.clusters,
            user_constraints_history=acc_constraints,
            weight_reasoning_history=weight_history,
        )
        await session.send({
            "type": "systems_analysis",
            "systems": [
                {
                    "cluster_id": s.cluster_id,
                    "suggested_name": s.suggested_name,
                    "description": s.description,
                    "build_or_buy": s.build_or_buy,
                    "build_or_buy_rationale": s.build_or_buy_rationale,
                    "market_options": [
                        {"name": o.name, "fit_rationale": o.fit_rationale}
                        for o in s.market_options
                    ],
                }
                for s in sa.systems
            ],
        })

        await session.send({
            "type": "status",
            "message": "Evaluating EA principles compliance…",
            "loading": "compliance",
        })
        comp = await gen.evaluate_ea_compliance(
            final_clusters=bsp.clusters,
            ea_principles=baseline,
            weight_reasoning_history=weight_history,
        )
        await session.send({
            "type": "ea_compliance",
            "evaluations": [
                {
                    "principle": ev.principle,
                    "status": ev.status,
                    "justification": ev.justification,
                }
                for ev in comp.evaluations
            ],
        })

        tobe_metrics = compute_isa_metrics(bsp.clusters, mat.matrix, mat.process_types)

        await session.send({
            "type": "question",
            "message": "Do you want to compare this proposed architecture with your current application landscape?",
            "choices": ["Yes — upload my current landscape", "No — finish here"],
            "inferred_constraints": [],
        })
        ans = await session.wait()

        if ans in ("yes", "y", "Yes — upload my current landscape"):
            await session.send({"type": "upload_as_is"})
            raw_csv = await session.wait()

            if raw_csv:
                content = base64.b64decode(raw_csv).decode("utf-8")
                as_is_mapping: dict = {}
                for row in csv.reader(io.StringIO(content)):
                    if len(row) >= 2:
                        proc, app = row[0].strip(), row[1].strip()
                        if proc and app:
                            as_is_mapping.setdefault(app, []).append(proc)

                asis_clusters = as_is_to_clusters(as_is_mapping, mat.matrix)
                asis_metrics  = compute_isa_metrics(asis_clusters, mat.matrix, mat.process_types)

                await session.send({
                    "type": "status",
                    "message": "Comparing As-Is vs. To-Be architecture…",
                    "loading": "comparison",
                })
                comparison = await gen.compare_as_is_to_be(
                    as_is_mapping=as_is_mapping,
                    final_clusters=bsp.clusters,
                    systems_analysis=sa,
                    asis_metrics=asis_metrics,
                    tobe_metrics=tobe_metrics,
                )
                await session.send({
                    "type": "as_is_comparison",
                    "asis_metrics": asis_metrics,
                    "tobe_metrics": tobe_metrics,
                    "overall_summary": comparison.overall_summary,
                    "estimated_complexity": comparison.estimated_complexity,
                    "complexity_rationale": comparison.complexity_rationale,
                    "alignments": [
                        {
                            "proposed_cluster_id": a.proposed_cluster_id,
                            "proposed_system_name": a.proposed_system_name,
                            "current_applications": a.current_applications,
                            "processes_to_acquire": a.processes_to_acquire,
                            "processes_to_release": a.processes_to_release,
                            "alignment_summary": a.alignment_summary,
                        }
                        for a in comparison.alignments
                    ],
                    "transformation_steps": [
                        {
                            "step_number": s.step_number,
                            "action": s.action,
                            "description": s.description,
                            "rationale": s.rationale,
                            "affected_processes": s.affected_processes,
                            "affected_applications": s.affected_applications,
                        }
                        for s in comparison.transformation_steps
                    ],
                })

        await session.send({"type": "complete", "message": "Architecture analysis complete!"})

    except Exception as e:
        await session.send({"type": "error", "message": str(e)})


@app.get("/")
async def root():
    return FileResponse(str(STATIC_DIR / "index.html"))


@app.websocket("/ws/{sid}")
async def ws_endpoint(websocket: WebSocket, sid: str):
    await websocket.accept()
    session = Session(websocket)
    sessions[sid] = session

    try:
        while True:
            raw = await websocket.receive_json()
            t = raw.get("type")

            if t == "start":
                if "csv_b64" in raw:
                    content = base64.b64decode(raw["csv_b64"]).decode("utf-8")
                    reader = csv.reader(io.StringIO(content))
                    session.processes = [row[0].strip() for row in reader if row and row[0].strip()]
                elif "processes_text" in raw:
                    session.processes = [p.strip() for p in raw["processes_text"].split(",") if p.strip()]

                if "pdf_b64" in raw:
                    pdf_bytes = base64.b64decode(raw["pdf_b64"])
                    text = ""
                    with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
                        for page in doc:
                            text += page.get_text()
                    session.context = text
                elif "txt_b64" in raw:
                    session.context = base64.b64decode(raw["txt_b64"]).decode("utf-8")
                elif "context_text" in raw:
                    session.context = raw["context_text"]

                asyncio.create_task(pipeline(session))

            elif t == "answer":
                await session.queue.put(raw.get("value", ""))

    except WebSocketDisconnect:
        sessions.pop(sid, None)

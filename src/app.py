import asyncio
import base64
import csv
import io
import json
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
from utils.bsp import run_bsp, compute_isa_metrics, matrix_to_clusters

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

        debug_path = Path(__file__).parent / "last_extraction_debug.json"
        debug_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
        print(f"Extraction dumped to {debug_path} — "
              f"{len(data.get('processes', []))} processes, "
              f"{len(data.get('entities', []))} entities, "
              f"{len(data.get('operations', []))} operations")

        inferred = [c["description"] for c in data.get("inferred_constraints", [])]
        if inferred:
            await session.send({"type": "constraints_found", "constraints": inferred})

        mat = Matrix()
        defined = {e["name"] for e in data.get("entities", [])}
        skipped_ops = []
        for p in data.get("processes", []):
            mat.set_process_type(p["name"], p["process_type"])
        for op in data.get("operations", []):
            if op["entity_name"] not in defined:
                skipped_ops.append(op)
                continue
            mat.add_entry(op["process_name"], op["entity_name"], op["operation"])

        if skipped_ops:
            print(f"  WARNING: {len(skipped_ops)} operation(s) skipped — "
                  f"entity_name didn't match any defined entity: "
                  f"{[(o['process_name'], o['entity_name']) for o in skipped_ops[:10]]}")

        processes_with_no_ops = [p["name"] for p in data.get("processes", []) if p["name"] not in mat.matrix]
        if processes_with_no_ops:
            print(f"  WARNING: {len(processes_with_no_ops)} process(es) got zero valid operations "
                  f"and will be missing from the matrix: {processes_with_no_ops}")

        data_quality_warnings = {
            "skipped_operations": [
                {"process_name": o["process_name"], "entity_name": o["entity_name"], "operation": o["operation"]}
                for o in skipped_ops
            ],
            "processes_missing_from_matrix": processes_with_no_ops,
        }
        if skipped_ops or processes_with_no_ops:
            await session.send({
                "type": "data_quality_warning",
                "message": (
                    f"{len(skipped_ops)} operation(s) were dropped (entity name mismatch) and "
                    f"{len(processes_with_no_ops)} process(es) have no valid operations and are "
                    f"missing from the matrix below."
                ),
                "skipped_operations": data_quality_warnings["skipped_operations"],
                "processes_missing_from_matrix": processes_with_no_ops,
            })

        df_raw = pd.DataFrame(mat.matrix).T.fillna("")
        await session.send({
            "type": "crud_matrix",
            "processes": list(df_raw.index),
            "entities": list(df_raw.columns),
            "data": df_raw.to_dict(),
        })

        await session.send({
            "type": "process_type_review",
            "processes": [
                {"name": p["name"], "process_type": p["process_type"]}
                for p in data.get("processes", [])
                if p["name"] in mat.matrix
            ],
        })
        reviewed = await session.wait()
        process_type_overrides = {}
        if isinstance(reviewed, dict):
            for name, ptype in reviewed.items():
                if name not in mat.process_types or ptype not in ("atomic", "end_to_end", "ambiguous"):
                    continue
                if ptype != mat.process_types[name]:
                    process_type_overrides[name] = {"from": mat.process_types[name], "to": ptype}
                mat.set_process_type(name, ptype)

        await session.send({"type": "status", "message": "Running BSP clustering…", "loading": "bsp"})
        bsp = run_bsp(mat.matrix, process_types=mat.process_types)
        initial_bsp = bsp
        bsp_iterations: list = []
        iteration_metrics = compute_isa_metrics(bsp.clusters, mat.matrix, mat.process_types)

        await session.send({
            "type": "clustering",
            "iteration": 0,
            "title": "Initial BSP Clustering",
            "clusters": [c.to_dict() for c in bsp.clusters],
            "matrix": _matrix_payload(bsp.reordered_matrix),
            "metrics": iteration_metrics,
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
            iteration_metrics = compute_isa_metrics(bsp.clusters, mat.matrix, mat.process_types)
            await session.send({
                "type": "clustering",
                "iteration": iteration,
                "title": f"BSP — Iteration {iteration}",
                "clusters": [c.to_dict() for c in bsp.clusters],
                "matrix": _matrix_payload(bsp.reordered_matrix),
                "metrics": iteration_metrics,
            })

            bsp_iterations.append({
                "iteration": iteration,
                "user_constraint": constraint,
                "ea_weights": {
                    str(k): {"weight": v, "reasoning": r.get(k, "")}
                    for k, v in w.items()
                },
                "clusters": [c.to_dict() for c in bsp.clusters],
                "entity_owners": bsp.entity_owners,
                "metrics": iteration_metrics,
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
        comparison = None
        asis_metrics = None

        await session.send({
            "type": "question",
            "message": "Do you want to compare this proposed architecture with your current application landscape?",
            "choices": ["Yes — upload my current landscape", "No — finish here"],
            "inferred_constraints": [],
        })
        ans = await session.wait()

        if ans in ("yes", "y", "Yes — upload my current landscape"):
            await session.send({
                "type": "upload_as_is",
                "format": "Process,Entity,Operation,System,ProcessType",
                "format_notes": (
                    "One row per CRUD entry in your CURRENT landscape's own CRUD matrix — independent "
                    "process/entity vocabulary, does not need to match the proposed architecture. "
                    "Operation is one of C/R/U/D. System is the current application/system that process "
                    "belongs to. ProcessType is one of atomic/end_to_end/ambiguous."
                ),
            })
            raw_csv = await session.wait()

            if raw_csv:
                content = base64.b64decode(raw_csv).decode("utf-8")
                as_is_matrix: dict = {}
                as_is_process_system: dict = {}
                as_is_process_types: dict = {}
                parse_errors = []

                rows = list(csv.reader(io.StringIO(content)))
                if rows and rows[0] and rows[0][0].strip().lower() == "process":
                    rows = rows[1:]

                for i, row in enumerate(rows, start=1):
                    if not row or not row[0].strip():
                        continue
                    if len(row) < 5:
                        parse_errors.append(f"row {i}: expected 5 columns (Process,Entity,Operation,System,ProcessType), got {len(row)}")
                        continue
                    proc, ent, op, system, ptype = (c.strip() for c in row[:5])
                    if not proc or not ent or not op or not system:
                        parse_errors.append(f"row {i}: Process, Entity, Operation, and System are all required")
                        continue
                    if ptype not in ("atomic", "end_to_end", "ambiguous"):
                        parse_errors.append(f"row {i}: ProcessType '{ptype}' for '{proc}' must be one of atomic/end_to_end/ambiguous")
                        continue
                    as_is_matrix.setdefault(proc, {})[ent] = op.upper()
                    as_is_process_system[proc] = system
                    as_is_process_types[proc] = ptype

                if parse_errors or not as_is_matrix:
                    await session.send({
                        "type": "error",
                        "message": (
                            "As-Is CSV could not be parsed — skipping the comparison. Expected columns: "
                            "Process,Entity,Operation,System,ProcessType (ProcessType one of "
                            "atomic/end_to_end/ambiguous). Issues:\n" + "\n".join(parse_errors[:20])
                        ),
                    })
                else:
                    as_is_clusters = matrix_to_clusters(as_is_matrix, as_is_process_system)
                    asis_metrics = compute_isa_metrics(as_is_clusters, as_is_matrix, as_is_process_types)

                    await session.send({
                        "type": "isa_metrics_comparison",
                        "asis_metrics": asis_metrics,
                        "tobe_metrics": tobe_metrics,
                    })

                    await session.send({
                        "type": "status",
                        "message": "Running As-Is vs. To-Be gap analysis…",
                        "loading": "comparison",
                    })
                    comparison = await gen.compare_as_is_to_be_matrices(
                        as_is_clusters=as_is_clusters,
                        to_be_clusters=bsp.clusters,
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

        log_data = {
            "extraction": data,
            "data_quality_warnings": data_quality_warnings,
            "process_type_overrides": process_type_overrides,
            "initial_bsp": {
                "clusters": [c.to_dict() for c in initial_bsp.clusters],
                "entity_owners": initial_bsp.entity_owners,
            },
            "bsp_iterations": bsp_iterations,
            "final_bsp": {
                "clusters": [c.to_dict() for c in bsp.clusters],
                "entity_owners": bsp.entity_owners,
                "ea_weights": {
                    str(k): {"weight": v, "reasoning": acc_reasoning.get(k, "")}
                    for k, v in acc_weights.items()
                },
                "user_constraints": acc_constraints.strip(),
            },
            "systems_analysis": [
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
            "ea_compliance": [
                {
                    "principle": ev.principle,
                    "status": ev.status,
                    "justification": ev.justification,
                }
                for ev in comp.evaluations
            ],
            "isa_metrics_tobe": tobe_metrics,
            "isa_metrics_asis": asis_metrics,
            "as_is_comparison": {
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
            } if comparison else None,
        }
        log_path = Path(__file__).parent / "log.json"
        log_path.write_text(json.dumps(log_data, indent=2), encoding="utf-8")
        print(f"Full run log written to {log_path}")

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

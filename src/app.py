import asyncio
import base64
import csv
import io
import json
import os
from datetime import datetime
from pathlib import Path

import fitz
import pandas as pd
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

import sys
sys.path.insert(0, str(Path(__file__).parent))  # so `utils` imports work when run from src/

from utils import Matrix
from utils.generator import Generator
from utils.bsp import run_bsp, compute_isa_metrics, matrix_to_clusters, classify_changes

app = FastAPI()

STATIC_DIR = Path(__file__).parent / "static"
STATIC_DIR.mkdir(exist_ok=True)

app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


# One browser connection: the WebSocket plus a queue for the user's answers
class Session:
    def __init__(self, ws: WebSocket):
        self.ws = ws
        self.queue: asyncio.Queue = asyncio.Queue()
        self.processes: list[str] = []
        self.context: str = ""

    async def send(self, msg: dict):
        await self.ws.send_json(msg)

    # Waits for the next answer sent by the frontend
    async def wait(self) -> str:
        return await self.queue.get()


sessions: dict[str, Session] = {}


# Decodes an uploaded CSV/TXT as UTF-8, falling back to Windows-1252 (Excel on Portuguese Windows)
def _decode_uploaded_text(raw_bytes: bytes) -> str:
    try:
        return raw_bytes.decode("utf-8-sig")
    except UnicodeDecodeError:
        return raw_bytes.decode("cp1252")


# Converts a matrix DataFrame into the JSON shape the frontend expects
def _matrix_payload(df: pd.DataFrame) -> dict:
    return {
        "processes": list(df.index),
        "entities": list(df.columns),
        "data": df.fillna("").to_dict(),
    }


# Saves each extracted matrix to src/resources/initial_matrices/ (same shape run_bsp expects)
def _save_initial_matrix(mat: Matrix) -> None:
    out_dir = Path(__file__).parent / "resources" / "initial_matrices"
    out_dir.mkdir(parents=True, exist_ok=True)

    n_procs = len(mat.matrix)
    n_ents = len({e for ents in mat.matrix.values() for e in ents})
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = out_dir / f"matrix_{timestamp}_{n_procs}p_{n_ents}e.json"

    out_path.write_text(json.dumps({
        "saved_at": datetime.now().isoformat(),
        "matrix": mat.matrix,
        "process_types": mat.process_types,
    }, indent=2), encoding="utf-8")
    print(f"Initial matrix saved to {out_path}")


# Full EA-GPT pipeline for one session: extraction -> review -> BSP + refinement -> analysis -> optional As-Is
async def pipeline(session: Session):
    try:
        gen = Generator()

        await session.send({
            "type": "status",
            "message": f"Extracting CRUD matrix — {len(session.processes)} processes, {len(session.context)} chars of context…",
            "loading": "matrix",
        })

        # Extraction: LLM builds the CRUD matrix from the processes and context
        result, extraction_issues, extraction_attempts = await gen.extract(context=session.context, process_list=session.processes)
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

        # Build the matrix, dropping operations that reference an undefined entity
        mat = Matrix()
        defined ={e["name"] for e in data.get("entities", [])}
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

        _save_initial_matrix(mat)

        data_quality_warnings = {
            "skipped_operations": [
                {"process_name": o["process_name"], "entity_name": o["entity_name"], "operation": o["operation"]}
                for o in skipped_ops
            ],
            "processes_missing_from_matrix": processes_with_no_ops,
            "extraction_issues": extraction_issues or [],
        }
        if skipped_ops or processes_with_no_ops or extraction_issues:
            extra = f" {len(extraction_issues)} extraction issue(s) remained even after retries." if extraction_issues else ""
            await session.send({
                "type": "data_quality_warning",
                "message": (
                    f"{len(skipped_ops)} operation(s) were dropped (entity name mismatch) and "
                    f"{len(processes_with_no_ops)} process(es) have no valid operations and are "
                    f"missing from the matrix below.{extra}"
                ),
                "skipped_operations": data_quality_warnings["skipped_operations"],
                "processes_missing_from_matrix": processes_with_no_ops,
                "extraction_issues": data_quality_warnings["extraction_issues"],
            })

        df_raw = pd.DataFrame(mat.matrix).T.fillna("")
        await session.send({
            "type": "crud_matrix",
            "processes": list(df_raw.index),
            "entities": list(df_raw.columns),
            "data": df_raw.to_dict(),
            "extraction_attempts": extraction_attempts,
        })

        # Architect reviews and can override each process type (atomic / end_to_end)
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

        principles_path = Path(__file__).parent / "resources" / "ea_principles.txt"
        baseline = principles_path.read_text(encoding="utf-8")

        # Constraints and weights accumulate across iterations
        acc_constraints = ""
        acc_process_weights: dict = {}
        acc_process_reasoning: dict = {}
        acc_entity_weights: dict = {}
        acc_entity_reasoning: dict = {}
        bsp_iterations: list = []

        # 1. Structural clustering from the data alone (no EA principles, no constraints), shown as the "before"
        await session.send({"type": "status", "message": "Running BSP clustering…", "loading": "bsp"})
        structural_draft = run_bsp(mat.matrix, process_types=mat.process_types)
        structural_metrics = compute_isa_metrics(structural_draft.clusters, mat.matrix, mat.process_types)
        await session.send({
            "type": "structural_preview",
            "title": "Structural Clustering (before EA principles)",
            "clusters": [c.to_dict() for c in structural_draft.clusters],
            "matrix": _matrix_payload(structural_draft.reordered_matrix),
            "metrics": structural_metrics,
            "process_span": structural_draft.process_span,
            "unallocated_reads": structural_draft.unallocated_reads,
        })

        # 2. EA principles only (no architect constraints yet) -> weights -> re-run BSP = "Initial BSP Clustering"
        await session.send({
            "type": "status", "message": "Applying EA principles to initial clustering…",
            "loading": "bsp", "marker": "principles",
        })
        pw0, pr0, ew0, er0 = await gen._compute_ea_weights(structural_draft.clusters, baseline, "")
        acc_process_weights.update(pw0)
        acc_process_reasoning.update(pr0)
        acc_entity_weights.update(ew0)
        acc_entity_reasoning.update(er0)

        bsp = run_bsp(
            mat.matrix, process_types=mat.process_types,
            process_weights=acc_process_weights, process_reasoning=acc_process_reasoning,
            entity_weights=acc_entity_weights, entity_reasoning=acc_entity_reasoning,
        )
        initial_bsp = bsp
        iteration_metrics = compute_isa_metrics(bsp.clusters, mat.matrix, mat.process_types)

        # What the principles changed vs. the structural clustering, shown in the "i" popup
        struct_names = {c.id: c.name for c in structural_draft.clusters}
        new_names = {c.id: c.name for c in bsp.clusters}
        principles_changes = classify_changes(structural_draft.clusters, bsp.clusters)
        principles_detail = {
            "process_weights": {
                " ↔ ".join(k): {"weight": v, "reasoning": pr0.get(k, "")}
                for k, v in pw0.items()
            },
            "entity_weights": {
                " ↔ ".join(k): {"weight": v, "reasoning": er0.get(k, "")}
                for k, v in ew0.items()
            },
            "changes": [
                {
                    "subject": ch.subject,
                    "from_cluster": struct_names.get(ch.from_cluster) if ch.from_cluster is not None else None,
                    "to_cluster": new_names.get(ch.to_cluster) if ch.to_cluster is not None else None,
                }
                for ch in principles_changes
            ],
        }

        await session.send({
            "type": "clustering",
            "iteration": 0,
            "title": "Initial BSP Clustering",
            "clusters": [c.to_dict() for c in bsp.clusters],
            "matrix": _matrix_payload(bsp.reordered_matrix),
            "metrics": iteration_metrics,
            "process_span": bsp.process_span,
            "unallocated_reads": bsp.unallocated_reads,
            "principles_detail": principles_detail,
        })

        acc_confirmed_overrides: dict[str, set] = {}
        acc_declined_overrides: dict[str, set] = {}
        acc_override_history: list[dict] = []  # full audit trail — why each override was decided
        iteration = 0

        # 3. Refinement loop: each constraint -> new weights -> re-run BSP, until the architect is satisfied
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

            # Clusters before this constraint, to show what it changed in the "i" popup
            prev_clusters = bsp.clusters

            await session.send({
                "type": "status",
                "message": "Analysing constraints and computing EA weights…",
                "loading": "weights",
                "marker": f"constraint-{iteration}",
            })
            pw, pr, ew, er = await gen._compute_ea_weights(bsp.clusters, baseline, acc_constraints)
            acc_process_weights.update(pw)
            acc_process_reasoning.update(pr)
            acc_entity_weights.update(ew)
            acc_entity_reasoning.update(er)

            trial = run_bsp(
                mat.matrix, process_weights=acc_process_weights, process_reasoning=acc_process_reasoning,
                process_types=mat.process_types, confirmed_overrides=acc_confirmed_overrides,
                entity_weights=acc_entity_weights, entity_reasoning=acc_entity_reasoning,
            )

            # New atomicity conflicts, grouped by process, to ask the architect about
            new_by_process: dict = {}
            for pc in trial.pending_conflicts:
                if pc.entity in acc_confirmed_overrides.get(pc.process, set()):
                    continue  # already confirmed in a prior iteration
                if pc.entity in acc_declined_overrides.get(pc.process, set()):
                    continue  # already declined — don't ask again
                new_by_process.setdefault(pc.process, []).append(pc)

            # Ask the architect to confirm or decline each conflict, then re-run with the confirmed ones
            if new_by_process:
                await session.send({
                    "type": "atomicity_override_confirm",
                    "candidates": [
                        {
                            "process": proc,
                            "conflicts": [
                                {
                                    "entity": pc.entity,
                                    "conflicting_process": pc.conflicting_process,
                                    "reasoning": pc.reasoning,
                                }
                                for pc in conflicts
                            ],
                        }
                        for proc, conflicts in new_by_process.items()
                    ],
                })
                decisions = await session.wait()
                if isinstance(decisions, dict):
                    for proc, entity_decisions in decisions.items():
                        if proc not in new_by_process or not isinstance(entity_decisions, dict):
                            continue
                        conflict_by_entity = {pc.entity: pc for pc in new_by_process[proc]}
                        for entity, accepted in entity_decisions.items():
                            target = acc_confirmed_overrides if accepted else acc_declined_overrides
                            target.setdefault(proc, set()).add(entity)

                            pc = conflict_by_entity.get(entity)
                            acc_override_history.append({
                                "iteration": iteration,
                                "process": proc,
                                "entity": entity,
                                "conflicting_process": pc.conflicting_process if pc else "",
                                "reasoning": pc.reasoning if pc else "",
                                "decision": "confirmed" if accepted else "declined",
                            })

                bsp = run_bsp(
                    mat.matrix, process_weights=acc_process_weights, process_reasoning=acc_process_reasoning,
                    process_types=mat.process_types, confirmed_overrides=acc_confirmed_overrides,
                    entity_weights=acc_entity_weights, entity_reasoning=acc_entity_reasoning,
                )
            else:
                bsp = trial

            prev_names = {c.id: c.name for c in prev_clusters}
            new_names = {c.id: c.name for c in bsp.clusters}
            constraint_changes = classify_changes(prev_clusters, bsp.clusters)
            constraint_detail = {
                "user_constraint": constraint,
                "process_weights": {
                    " ↔ ".join(k): {"weight": v, "reasoning": pr.get(k, "")}
                    for k, v in pw.items()
                },
                "entity_weights": {
                    " ↔ ".join(k): {"weight": v, "reasoning": er.get(k, "")}
                    for k, v in ew.items()
                },
                "changes": [
                    {
                        "subject": ch.subject,
                        "from_cluster": prev_names.get(ch.from_cluster) if ch.from_cluster is not None else None,
                        "to_cluster": new_names.get(ch.to_cluster) if ch.to_cluster is not None else None,
                    }
                    for ch in constraint_changes
                ],
            }

            iteration_metrics = compute_isa_metrics(bsp.clusters, mat.matrix, mat.process_types)
            await session.send({
                "type": "clustering",
                "iteration": iteration,
                "title": f"BSP — Iteration {iteration}",
                "clusters": [c.to_dict() for c in bsp.clusters],
                "matrix": _matrix_payload(bsp.reordered_matrix),
                "metrics": iteration_metrics,
                "process_span": bsp.process_span,
                "unallocated_reads": bsp.unallocated_reads,
                "constraint_detail": constraint_detail,
            })

            bsp_iterations.append({
                "iteration": iteration,
                "user_constraint": constraint,
                "process_weights": {
                    str(k): {"weight": v, "reasoning": pr.get(k, "")}
                    for k, v in pw.items()
                },
                "entity_weights": {
                    str(k): {"weight": v, "reasoning": er.get(k, "")}
                    for k, v in ew.items()
                },
                "atomicity_overrides": {
                    "confirmed": {p: sorted(es) for p, es in acc_confirmed_overrides.items()},
                    "declined": {p: sorted(es) for p, es in acc_declined_overrides.items()},
                    "still_pending": [
                        {"process": pc.process, "entity": pc.entity,
                         "conflicting_process": pc.conflicting_process, "reasoning": pc.reasoning}
                        for pc in bsp.pending_conflicts
                    ],
                    "resolution_history": list(acc_override_history),
                },
                "clusters": [c.to_dict() for c in bsp.clusters],
                "entity_owners": bsp.entity_owners,
                "process_span": bsp.process_span,
                "unallocated_reads": bsp.unallocated_reads,
                "metrics": iteration_metrics,
            })

        # Weight history as text, given to the LLM so it knows why the clusters look the way they do
        wrl = [
            f"{k[0]} ↔ {k[1]}: {acc_process_weights[k]:+.2f} — {acc_process_reasoning.get(k, '')}"
            for k in acc_process_reasoning
        ]
        wrl += [
            f"{k[0]} ↔ {k[1]} (entities): {acc_entity_weights[k]:+.2f} — {acc_entity_reasoning.get(k, '')}"
            for k in acc_entity_reasoning
        ]
        weight_history = "\n".join(wrl)

        # 4. Systems analysis
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

        # 5. EA compliance
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

        # 6. Optional As-Is vs. To-Be comparison (exploratory, outside the thesis scope)
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
                "format": "Process,Entity,System",
                "format_notes": (
                    "One row per (process, information/entity it touches, current application). "
                    "A real current landscape rarely has its own CRUD-level matrix — most "
                    "organisations can say which application supports which process and roughly "
                    "what information it touches, not the exact Create/Read/Update/Delete "
                    "breakdown, so this format doesn't ask for it. Process must match a name from "
                    "the proposed architecture's own process list (its process type is reused from "
                    "there); Entity is free text describing the information that process/application "
                    "touches — it does not need to match the proposed architecture's entity "
                    "vocabulary. System is the current application/system that process belongs to."
                ),
            })
            raw_csv = await session.wait()

            if raw_csv:
                content = _decode_uploaded_text(base64.b64decode(raw_csv))
                as_is_matrix: dict = {}
                as_is_process_system: dict = {}
                as_is_process_types: dict = {}
                parse_errors = []
                unknown_processes = set()

                rows = list(csv.reader(io.StringIO(content)))
                if rows and rows[0] and rows[0][0].strip().lower() == "process":
                    rows = rows[1:]

                for i, row in enumerate(rows, start=1):
                    if not row or not row[0].strip():
                        continue
                    if len(row) < 3:
                        parse_errors.append(f"row {i}: expected 3 columns (Process,Entity,System), got {len(row)}")
                        continue
                    proc, ent, system = (c.strip() for c in row[:3])
                    if not proc or not ent or not system:
                        parse_errors.append(f"row {i}: Process, Entity, and System are all required")
                        continue
                    # The As-Is format has no CRUD letters, so every touch is treated as a write
                    as_is_matrix.setdefault(proc, {})[ent] = "C"
                    as_is_process_system[proc] = system
                    # Reuse the process type from the extraction (unknown processes default to atomic)
                    if proc in mat.process_types:
                        as_is_process_types[proc] = mat.process_types[proc]
                    else:
                        unknown_processes.add(proc)
                        as_is_process_types[proc] = "atomic"

                if unknown_processes:
                    await session.send({
                        "type": "status",
                        "message": (
                            "Note: process name(s) not found in the proposed architecture's own "
                            "process list, so their ProcessType defaulted to atomic: "
                            + ", ".join(sorted(unknown_processes)[:20])
                        ),
                    })

                if parse_errors or not as_is_matrix:
                    await session.send({
                        "type": "error",
                        "message": (
                            "As-Is CSV could not be parsed — skipping the comparison. Expected columns: "
                            "Process,Entity,System. Issues:\n" + "\n".join(parse_errors[:20])
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

        # Write the whole session to src/log.json
        log_data = {
            "extraction": data,
            "extraction_attempts": extraction_attempts,
            "data_quality_warnings": data_quality_warnings,
            "process_type_overrides": process_type_overrides,
            "initial_bsp": {
                "clusters": [c.to_dict() for c in initial_bsp.clusters],
                "entity_owners": initial_bsp.entity_owners,
                "process_span": initial_bsp.process_span,
                "unallocated_reads": initial_bsp.unallocated_reads,
                "principles_detail": principles_detail,
            },
            "structural_draft": {
                "clusters": [c.to_dict() for c in structural_draft.clusters],
                "entity_owners": structural_draft.entity_owners,
                "process_span": structural_draft.process_span,
                "unallocated_reads": structural_draft.unallocated_reads,
            },
            "bsp_iterations": bsp_iterations,
            "final_bsp": {
                "clusters": [c.to_dict() for c in bsp.clusters],
                "entity_owners": bsp.entity_owners,
                "process_span": bsp.process_span,
                "unallocated_reads": bsp.unallocated_reads,
                "process_weights": {
                    str(k): {"weight": v, "reasoning": acc_process_reasoning.get(k, "")}
                    for k, v in acc_process_weights.items()
                },
                "entity_weights": {
                    str(k): {"weight": v, "reasoning": acc_entity_reasoning.get(k, "")}
                    for k, v in acc_entity_weights.items()
                },
                "atomicity_overrides": {
                    "confirmed": {p: sorted(es) for p, es in acc_confirmed_overrides.items()},
                    "declined": {p: sorted(es) for p, es in acc_declined_overrides.items()},
                    "still_pending": [
                        {"process": pc.process, "entity": pc.entity,
                         "conflicting_process": pc.conflicting_process, "reasoning": pc.reasoning}
                        for pc in bsp.pending_conflicts
                    ],
                    "resolution_history": list(acc_override_history),
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


# Serves the frontend
@app.get("/")
async def root():
    return FileResponse(str(STATIC_DIR / "index.html"))


# WebSocket: "start" reads the uploaded processes/context and launches the pipeline; "answer" feeds the queue
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
                    content = _decode_uploaded_text(base64.b64decode(raw["csv_b64"]))
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
                    session.context = _decode_uploaded_text(base64.b64decode(raw["txt_b64"]))
                elif "context_text" in raw:
                    session.context = raw["context_text"]

                asyncio.create_task(pipeline(session))

            elif t == "answer":
                await session.queue.put(raw.get("value", ""))

    except WebSocketDisconnect:
        sessions.pop(sid, None)

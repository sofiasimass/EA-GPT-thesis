import os
import json
import asyncio
import logging
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
from typing import List

from .bsp import Cluster, ProcessWeight, EntityWeight, ConstraintWeightsResult, SystemsAnalysisResult, MarketOption, EAComplianceResult, AsIsToBeResult
from .matrix import MatrixResult #output schema

load_dotenv()


def _validate_extraction(data: dict, process_list: List[str]) -> List[str]:
    """
    Checks the extraction result against the hard rules stated in prompt.txt
    (completeness, entity count, orphan entities, entities never created,
    dangling references, per-process density, combined-cell depth,
    cross-entity writes, and cross-process entity sharing).
    Returns a list of human-readable issue descriptions — empty if all checks pass.
    """
    issues = []

    input_names = {p.strip() for p in process_list}
    output_names = {p["name"] for p in data.get("processes", [])}

    missing = input_names - output_names
    if missing:
        sample = sorted(missing)[:10]
        issues.append(
            f"{len(missing)} process(es) from the input list are missing from the output: {sample}"
            + ("..." if len(missing) > 10 else "")
        )

    extra = output_names - input_names
    if extra:
        issues.append(f"{len(extra)} process(es) in the output don't match any input process name: {sorted(extra)[:10]}")

    # Entity count scales with the process list — a fixed absolute range (this used to be a
    # hard "10 to 20") miscalibrates for anything far from that size: it forces fabricated
    # entities on a small process list and forces real distinct entities to be collapsed
    # together on a large one. These bounds are generous on purpose — a sanity check against
    # gross under/over-differentiation, not a target to hit exactly, matching prompt.txt's
    # own "roughly one entity per 4-6 processes" guidance.
    n_entities = len(data.get("entities", []))
    n_processes = len(input_names)
    min_entities = max(2, round(n_processes * 0.10))
    max_entities = max(4, round(n_processes * 0.40))
    if not (min_entities <= n_entities <= max_entities):
        issues.append(
            f"entity count is {n_entities} for {n_processes} processes — expected roughly "
            f"{min_entities}-{max_entities} (about one entity per 4-6 processes)"
        )

    entity_names = {e["name"] for e in data.get("entities", [])}
    touched_entities = {op["entity_name"] for op in data.get("operations", [])}
    orphans = entity_names - touched_entities
    if orphans:
        issues.append(f"{len(orphans)} entit(y/ies) defined but never used in any operation: {sorted(orphans)}")

    # Every entity should be CREATED by at least one process — otherwise its data has no
    # origin anywhere in this process landscape. Either the entity is wrong (merge it into
    # whatever creates equivalent data) or the process that actually produces it is missing
    # its Create operation. Same CRUD-matrix-anomaly literature already cited in bsp.py's hub
    # detection: Bureš, Cerny, Frajtak & Ahmed, "Testing the Consistency of Business Data
    # Objects Using Extended Static Testing of CRUD Matrices," Cluster Computing 22(S4), 2019.
    #
    # Scoped to entity_names (properly defined entities) so a hallucinated entity reference
    # isn't flagged here too — that's the dangling-reference check's job, below, and doubling
    # up would give confusing, overlapping guidance for what's really one problem.
    created_entities = {
        op["entity_name"] for op in data.get("operations", [])
        if "C" in str(op.get("operation", "")).upper()
    }
    never_created = sorted((touched_entities & entity_names) - created_entities)
    if never_created:
        issues.append(
            f"{len(never_created)} entit(y/ies) are read/updated but never created by any "
            f"process — every entity needs a Create somewhere: {never_created[:10]}"
        )

    op_process_names = {op["process_name"] for op in data.get("operations", [])}
    unknown_op_processes = op_process_names - output_names
    if unknown_op_processes:
        issues.append(f"{len(unknown_op_processes)} operation(s) reference a process not in the output's process list: {sorted(unknown_op_processes)[:10]}")

    unknown_op_entities = touched_entities - entity_names
    if unknown_op_entities:
        issues.append(f"{len(unknown_op_entities)} operation(s) reference an entity not in the output's entity list: {sorted(unknown_op_entities)[:10]}")

    processes_with_no_ops = output_names - op_process_names
    if processes_with_no_ops:
        issues.append(f"{len(processes_with_no_ops)} process(es) have zero operations and would vanish from the matrix: {sorted(processes_with_no_ops)[:10]}")

    # Density: prompt.txt requires 4-8 CRUD entries per process. Too few gives BSP almost
    # nothing to cluster on for that process; too many usually means padding with weak reads.
    #
    # Briefly disabled (2026-09-01) as a diagnostic experiment after tracing this floor to a
    # real harm — pushing the LLM to reach 4 entries reliably produces generic, non-specific
    # padding reads, and those padded reads mathematically inflate weighted-Jaccard entity
    # similarity enough to merge unrelated entities (confirmed: Training Course/Uniform at
    # 0.94, Circuit/Daily Scale at 0.87 — pure padding artifacts, not real structure).
    # Re-enabled after the experiment: removing the floor entirely reproduced the exact
    # degenerate failure mode this check was added to catch in the first place (v0.4.10) —
    # every one of 62 processes collapsed to a single operation, far less clustering signal
    # than the padding version had. The floor is necessary; the padding it also causes is a
    # separate, still-open tension, not a reason to remove it.
    op_counts: dict[str, int] = {}
    for op in data.get("operations", []):
        op_counts[op["process_name"]] = op_counts.get(op["process_name"], 0) + 1

    too_sparse = sorted(p for p, n in op_counts.items() if 0 < n < 4)
    if too_sparse:
        issues.append(
            f"{len(too_sparse)} process(es) have fewer than the required 4 CRUD entries "
            f"(density requirement): {too_sparse[:10]}"
        )

    too_dense = sorted(p for p, n in op_counts.items() if n > 8)
    if too_dense:
        issues.append(
            f"{len(too_dense)} process(es) have more than the allowed 8 CRUD entries "
            f"(density requirement): {too_dense[:10]}"
        )

    # A process satisfying its 4-6 entry minimum by touching 4-6 DIFFERENT entities once each
    # is the "easy" way out — it never has to reason about a process's full lifecycle on one
    # entity (create it, then later read/update/delete that SAME record). A process that owns
    # an entity should show that as a combined cell (e.g. C+U on the same entity), per
    # prompt.txt's own "Manage Clients" example. Flagged at the extraction level, not per
    # process, because not every process needs one (a pure reader legitimately never will) —
    # zero anywhere across a real-sized extraction means density is being reached by breadth,
    # not depth, which starves entity-ownership resolution and clustering of a real signal.
    cell_op_counts: dict[tuple[str, str], int] = {}
    for op in data.get("operations", []):
        key = (op["process_name"], op["entity_name"])
        cell_op_counts[key] = cell_op_counts.get(key, 0) + 1
    multi_op_cells = sum(1 for n in cell_op_counts.values() if n > 1)
    if len(output_names) >= 5 and multi_op_cells == 0:
        issues.append(
            "no (process, entity) pair anywhere has more than one operation — every process "
            "reached its operation count by touching several DIFFERENT entities once each, "
            "never by performing more than one operation on the SAME entity. Revisit processes "
            "that plausibly own an entity's lifecycle (creating it, then later reading/updating/"
            "deleting that SAME record) and consolidate onto that one entity instead of spreading "
            "across several."
        )

    # A process that writes (C/U/D) to two DIFFERENT entities in its own transaction is the
    # only extraction-level signal bsp.py's _group_entities ever gets for "these two entities
    # belong together" that isn't diluted through a chain of shared reads (see prompt.txt's
    # "Manage Client"/library worked examples, and its own checklist question: "Does this
    # process's own transaction genuinely change a SECOND, different entity, not just its main
    # one?"). That question is asked per process, one at a time — nothing checks the aggregate
    # outcome. Confirmed live (2026-09-11): a real 62-process extraction, after 3 full retry
    # attempts, ended with EVERY single writing process owning exactly one entity and reading
    # the rest — zero processes anywhere with cross-entity writes — despite the prompt asking
    # the question 62 times over. Flagged at the extraction level, not per process (a process
    # that legitimately only ever touches one entity is normal and fine on its own), because
    # zero anywhere across a real-sized extraction means entity clustering has to fall back
    # entirely on shared-read similarity, which weighted-Jaccard treats as proportionally as
    # strong as a shared write (R=1 vs C=4 is a ratio, not an absolute gap) — exactly the
    # padding-driven false-merge failure mode traced earlier the same day (Training Course/
    # Uniform at 0.94 similarity from padding reads alone).
    cud_entities_by_process: dict[str, set] = {}
    for op in data.get("operations", []):
        if str(op.get("operation", "")).upper() in ("C", "U", "D"):
            cud_entities_by_process.setdefault(op["process_name"], set()).add(op["entity_name"])
    cross_entity_writers = sum(1 for ents in cud_entities_by_process.values() if len(ents) > 1)
    if len(output_names) >= 5 and cross_entity_writers == 0:
        issues.append(
            "no process anywhere creates/updates/deletes more than one entity — every writing "
            "process owns exactly one entity and only reads the rest, so entity clustering has "
            "no direct write-based evidence that any two entities belong together. Revisit "
            "processes whose own transaction plausibly changes a second record as a side effect "
            "(e.g. closing something out while freeing/updating what it depended on) and record "
            "that second change as a real C/U/D, not a plain READ."
        )

    # Entities are supposed to be SHARED data objects (prompt.txt's ENTITIES section) — one
    # touched by only a single process gives BSP clustering no cross-process signal at all.
    # A handful is normal, but if most entities end up this way the extraction likely fell
    # back to a private entity per process instead of finding the real overlaps.
    entity_process_counts: dict[str, set] = {}
    for op in data.get("operations", []):
        entity_process_counts.setdefault(op["entity_name"], set()).add(op["process_name"])
    single_process_entities = sorted(e for e, procs in entity_process_counts.items() if len(procs) < 2)
    if entity_names and len(single_process_entities) > len(entity_names) / 2:
        issues.append(
            f"{len(single_process_entities)} of {len(entity_names)} entities are touched by only one "
            f"process each — entities should be shared data objects genuinely touched by multiple "
            f"processes, not a private entity per process: {single_process_entities[:10]}"
        )

    return issues


def _write_validation_debug(debug_info: dict) -> None:
    """
    Dumps _validate_extraction's findings across the extraction attempt(s)
    to a file, not just console prints — console output only reaches
    whoever's watching the terminal live, and isn't something that can be
    read back afterward (e.g. to debug a run from its log). Mirrors
    app.py's last_extraction_debug.json, same directory, so both debug
    files are found in the same place.
    """
    debug_path = Path(__file__).parent.parent / "last_extraction_validation_debug.json"
    debug_path.write_text(json.dumps(debug_info, indent=2), encoding="utf-8")


class Generator:
    def __init__(self, model_name: str = "gpt-4o"):

        # temperature=0 minimises but does not guarantee determinism — OpenAI's own docs
        # note residual backend non-determinism (batching, floating-point non-associativity)
        # even at temperature 0. `seed` is their "best effort" mechanism to reduce that
        # further; still not an absolute guarantee, but meaningfully more reproducible run
        # to run, which matters here since prompt-change comparisons assume "same input in,
        # same output out" so a difference is attributable to the prompt, not run-to-run luck.
        self.llm = ChatOpenAI(model=model_name, temperature=0, seed=42, request_timeout=120)
        self.structured_output = self.llm.with_structured_output(MatrixResult)

    def _resolve_biases(self, biases, canonical: dict, attr_a: str, attr_b: str, label: str):
        """
        Shared name-resolution loop for both axes: validates the LLM's names
        against the canonical (real) name list, skips anything hallucinated,
        and builds the weight_map/reasoning_map keyed by sorted tuples — the
        same shape bsp.py's _affinity / _group_processes / _group_entities
        look up by. attr_a/attr_b are the Pydantic field names to read
        (first_process/second_process for ProcessWeight, first_entity/
        second_entity for EntityWeight) since the two schemas share this
        exact same shape otherwise.
        """
        weight_map = {}
        reasoning_map = {}
        skipped = []
        for bias in biases:
            name_a = getattr(bias, attr_a).strip().lower()
            name_b = getattr(bias, attr_b).strip().lower()

            resolved_a = canonical.get(name_a)
            resolved_b = canonical.get(name_b)

            if resolved_a is None or resolved_b is None:
                # LLM hallucinated a name — skip so it doesn't silently zero-out.
                skipped.append((getattr(bias, attr_a), getattr(bias, attr_b)))
                continue

            # Keys must be sorted so (A,B) == (B,A) everywhere in bsp.py.
            pair = tuple(sorted([resolved_a, resolved_b]))
            weight_map[pair] = bias.weight
            reasoning_map[pair] = bias.reasoning
            print(f"  {label} Bias: {pair} -> {bias.weight:+.2f}  ({bias.reasoning})")

        if skipped:
            print(f"  WARNING: {len(skipped)} {label.lower()} bias(es) skipped — LLM used unrecognised names: {skipped}")

        return weight_map, reasoning_map

    async def _compute_ea_weights(self, current_clusters: List[Cluster], ea_principles: str, user_constraints: str = ""):
        """
        Analyzes the current BSP result and returns weights on two axes:
        process pairs (feeds bsp.py's _group_processes) and entity pairs
        (feeds _group_entities directly) — see bsp.py's module docstring
        for why both exist.

        Returns (process_weights, process_reasoning, entity_weights, entity_reasoning).
        Keys are always sorted tuples of the original names so they match
        the lookup keys built inside bsp.py.
        """
        # Build canonical name maps: lowercase-stripped → original name.
        # This is the source of truth used to validate names the LLM returns.
        all_process_names: List[str] = []
        all_entity_names: List[str] = []
        for c in current_clusters:
            all_process_names.extend(c.processes)
            all_entity_names.extend(c.entities)
        canonical_procs: dict[str, str] = {p.strip().lower(): p for p in all_process_names}
        canonical_ents: dict[str, str] = {e.strip().lower(): e for e in all_entity_names}

        cluster_summary = "\n".join([c.summary() for c in current_clusters])

        # Embed the exact process/entity names in the prompt so the LLM cannot drift.
        exact_names_list = "\n".join(f"  - {p}" for p in all_process_names)
        exact_entity_names_list = "\n".join(f"  - {e}" for e in all_entity_names)

        prompt_path = os.path.join(os.path.dirname(__file__), "..", "resources", "bsp_prompt.txt")
        with open(prompt_path, "r", encoding="utf-8") as f:
            base_instructions = f.read()
        prompt = ChatPromptTemplate.from_template(base_instructions + "\n\n")

        weight_llm = self.llm.with_structured_output(ConstraintWeightsResult)

        # Let exceptions propagate — a silent {} hides real problems and
        # makes the final matrix identical to the initial one.
        response = await (prompt | weight_llm).ainvoke({
            "ea_principles": ea_principles,
            "user_constraints": user_constraints if user_constraints.strip() else "(none provided)",
            "cluster_summary": cluster_summary,
            "exact_names_list": exact_names_list,
            "exact_entity_names_list": exact_entity_names_list,
        })

        process_weights, process_reasoning = self._resolve_biases(
            response.process_biases, canonical_procs, "first_process", "second_process", "Process",
        )
        entity_weights, entity_reasoning = self._resolve_biases(
            response.entity_biases, canonical_ents, "first_entity", "second_entity", "Entity",
        )

        if not process_weights and not entity_weights:
            print("  WARNING: No valid weights were produced on either axis. "
                  "The final matrix will be identical to the initial one.")
        else:
            print(f"  {len(process_weights)} process weight(s), {len(entity_weights)} entity weight(s) applied successfully.")

        return process_weights, process_reasoning, entity_weights, entity_reasoning

    async def describe_systems(
        self,
        final_clusters: List[Cluster],
        user_constraints_history: str,
        weight_reasoning_history: str,
    ) -> SystemsAnalysisResult:
        """
        Generates a business description and market alternative suggestions for each
        final BSP cluster. Reads the full decision trail (user constraints across all
        iterations and the reasoning behind each EA weight change) so the LLM
        understands *why* the systems ended up as they did, not just what they contain.
        """
        cluster_details = "\n\n".join(
            f"Cluster {c.id}:\n"
            f"  Processes : {', '.join(c.processes) or '—'}\n"
            f"  Entities  : {', '.join(c.entities) or '—'}"
            for c in final_clusters
        )

        prompt_path = os.path.join(os.path.dirname(__file__), "..", "resources", "description_prompt.txt")
        with open(prompt_path, "r", encoding="utf-8") as f:
            base_instructions = f.read()
        prompt = ChatPromptTemplate.from_template(base_instructions + "\n\n")

        systems_llm = self.llm.with_structured_output(SystemsAnalysisResult)
        return await (prompt | systems_llm).ainvoke({
            "cluster_details": cluster_details,
            "user_constraints_history": user_constraints_history.strip() if user_constraints_history.strip() else "(no iterations — initial clustering accepted)",
            "weight_reasoning_history": weight_reasoning_history.strip() if weight_reasoning_history.strip() else "(no weight adjustments made)",
        })

    async def evaluate_ea_compliance(
        self,
        final_clusters: List[Cluster],
        ea_principles: str,
        weight_reasoning_history: str,
    ) -> EAComplianceResult:
        """
        Evaluates whether the final BSP clustering respects each EA principle.
        Returns one verdict per principle with a justification grounded in the
        specific clusters, processes, and entities of this exercise.
        """
        cluster_details = "\n\n".join(
            f"Cluster {c.id}:\n"
            f"  Processes : {', '.join(c.processes) or '—'}\n"
            f"  Entities  : {', '.join(c.entities) or '—'}"
            for c in final_clusters
        )

        prompt_path = os.path.join(os.path.dirname(__file__), "..", "resources", "compliance_prompt.txt")
        with open(prompt_path, "r", encoding="utf-8") as f:
            base_instructions = f.read()
        prompt = ChatPromptTemplate.from_template(base_instructions + "\n\n")

        compliance_llm = self.llm.with_structured_output(EAComplianceResult)
        return await (prompt | compliance_llm).ainvoke({
            "cluster_details": cluster_details,
            "ea_principles": ea_principles,
            "weight_reasoning_history": weight_reasoning_history.strip() if weight_reasoning_history.strip() else "(no weight adjustments made)",
        })

    async def compare_as_is_to_be(
        self,
        as_is_mapping: dict,
        final_clusters: List[Cluster],
        systems_analysis: SystemsAnalysisResult,
        asis_metrics: dict,
        tobe_metrics: dict,
    ) -> AsIsToBeResult:
        """
        Compares the user's current application landscape (As-Is) against the
        proposed BSP clustering (To-Be) and returns a gap analysis with
        ISA metric comparison and ordered migration steps.
        """
        proposed_lines = []
        for s in systems_analysis.systems:
            cluster = next((c for c in final_clusters if c.id == s.cluster_id), None)
            if cluster:
                proposed_lines.append(
                    f"System {s.cluster_id}: {s.suggested_name}\n"
                    f"  Processes  : {', '.join(cluster.processes) or '—'}\n"
                    f"  Entities   : {', '.join(cluster.entities) or '—'}\n"
                    f"  Description: {s.description}"
                )
        proposed_systems = "\n\n".join(proposed_lines)

        as_is_lines = [f"{app}: {', '.join(procs)}" for app, procs in as_is_mapping.items()]
        as_is_formatted = "\n".join(as_is_lines)

        metric_meta = {
            "RSF":    ("1.0", "avg IS blocks per process — 1 = min coupling"),
            "NAIEF":  ("1.0", "entity write authority — 1 = single source of truth"),
            "LCOISF": ("1.0", "cluster cohesion — 1 = no giant-stain clusters"),
            "CPSMF":  ("1.0", "critical/non-critical isolation — 1 = perfect separation"),
        }
        rows = ["Metric  | As-Is | To-Be | Ideal | Meaning",
                "--------|-------|-------|-------|--------"]
        for m, (ideal, desc) in metric_meta.items():
            rows.append(f"{m:7} | {str(asis_metrics.get(m,'N/A')):5} | {str(tobe_metrics.get(m,'N/A')):5} | {ideal:5} | {desc}")
        metrics_comparison = "\n".join(rows)

        prompt_path = os.path.join(os.path.dirname(__file__), "..", "resources", "comparison_prompt.txt")
        with open(prompt_path, "r", encoding="utf-8") as f:
            base_instructions = f.read()
        prompt = ChatPromptTemplate.from_template(base_instructions + "\n\n")

        comparison_llm = self.llm.with_structured_output(AsIsToBeResult)
        return await (prompt | comparison_llm).ainvoke({
            "proposed_systems":    proposed_systems,
            "as_is_mapping":       as_is_formatted,
            "metrics_comparison":  metrics_comparison,
        })

    async def compare_as_is_to_be_matrices(
        self,
        as_is_clusters: List[Cluster],
        to_be_clusters: List[Cluster],
        systems_analysis: SystemsAnalysisResult,
        asis_metrics: dict,
        tobe_metrics: dict,
    ) -> AsIsToBeResult:
        """
        Gap analysis between two INDEPENDENTLY modeled architectures (own As-Is
        CRUD matrix, own vocabulary). Reasons about correspondence at the system
        level only — each system's name plus its own processes/entities as
        evidence of what it does — rather than trying to reconcile individual
        As-Is/To-Be process or entity names.
        """
        proposed_lines = []
        for s in systems_analysis.systems:
            cluster = next((c for c in to_be_clusters if c.id == s.cluster_id), None)
            if cluster:
                proposed_lines.append(
                    f"System {s.cluster_id}: {s.suggested_name}\n"
                    f"  Processes  : {', '.join(cluster.processes) or '—'}\n"
                    f"  Entities   : {', '.join(cluster.entities) or '—'}\n"
                    f"  Description: {s.description}"
                )
        proposed_systems = "\n\n".join(proposed_lines)

        as_is_systems = "\n\n".join(
            f"System {c.id}: {c.name}\n"
            f"  Processes : {', '.join(c.processes) or '—'}\n"
            f"  Entities  : {', '.join(c.entities) or '—'}"
            for c in as_is_clusters
        )

        metric_meta = {
            "RSF":    ("1.0", "avg IS blocks per process — 1 = min coupling"),
            "NAIEF":  ("1.0", "entity write authority — 1 = single source of truth"),
            "LCOISF": ("1.0", "cluster cohesion — 1 = no giant-stain clusters"),
            "CPSMF":  ("1.0", "critical/non-critical isolation — 1 = perfect separation"),
        }
        rows = ["Metric  | As-Is | To-Be | Ideal | Meaning",
                "--------|-------|-------|-------|--------"]
        for m, (ideal, desc) in metric_meta.items():
            rows.append(f"{m:7} | {str(asis_metrics.get(m,'N/A')):5} | {str(tobe_metrics.get(m,'N/A')):5} | {ideal:5} | {desc}")
        metrics_comparison = "\n".join(rows)

        prompt_path = os.path.join(os.path.dirname(__file__), "..", "resources", "comparison_matrices_prompt.txt")
        with open(prompt_path, "r", encoding="utf-8") as f:
            base_instructions = f.read()
        prompt = ChatPromptTemplate.from_template(base_instructions + "\n\n")

        comparison_llm = self.llm.with_structured_output(AsIsToBeResult)
        return await (prompt | comparison_llm).ainvoke({
            "proposed_systems":   proposed_systems,
            "as_is_systems":      as_is_systems,
            "metrics_comparison": metrics_comparison,
        })

    async def extract(self, context: str, process_list: list, max_attempts: int = 3):
        """
        Retries against _validate_extraction up to max_attempts times, not
        just once — a single retry kept leaving real issues unresolved
        (confirmed live, three separate times: an atomic-only, an
        end_to_end-only, and a mixed-type run each had at least one
        never_created/density/entity-sharing violation survive the one
        retry that used to exist). Unbounded retrying isn't safe either —
        nothing guarantees the model ever satisfies every check at once,
        so this stops after max_attempts and keeps whichever attempt had
        the FEWEST remaining issues, not necessarily the last one (a later
        attempt can regress on something an earlier one got right).

        Returns (result, remaining_issues, attempts_log) instead of just
        result — attempts_log is the full per-attempt issue history (not
        just the final one), so app.py can show the user the same
        before/after transparency the EA-principles "i" popup already
        gives for clustering, applied here to the retry loop instead.
        """
        prompt_path = os.path.join(os.path.dirname(__file__), "..", "resources", "prompt.txt")
        with open(prompt_path, "r", encoding="utf-8") as f:
            base_instructions = f.read()

        process_list_str = "\n".join(f"  - {p}" for p in process_list)

        base_prompt = ChatPromptTemplate.from_template(
            base_instructions + "\n"
            "TEXT (may be empty — see \"WHEN NO CONTEXT IS PROVIDED\" above):\n{context}\n"
            "Extracted CRUD Matrix:"
        )
        retry_prompt = ChatPromptTemplate.from_template(
            base_instructions + "\n"
            "TEXT (may be empty — see \"WHEN NO CONTEXT IS PROVIDED\" above):\n{context}\n"
            "\n"
            "IMPORTANT — YOUR PREVIOUS ATTEMPT VIOLATED THESE REQUIREMENTS:\n{issues}\n"
            "Fix every one of these specific problems in this attempt. In particular, re-check "
            "that your \"processes\" output array has exactly one entry per process listed above, "
            "and that every operation's process_name/entity_name matches names you actually defined.\n"
            "Extracted CRUD Matrix:"
        )

        best_result, best_issues = None, None
        attempts_log = []
        issues_str = ""

        for attempt in range(1, max_attempts + 1):
            if attempt == 1:
                result = await (base_prompt | self.structured_output).ainvoke({
                    "process_list": process_list_str,
                    "context": context,
                })
            else:
                result = await (retry_prompt | self.structured_output).ainvoke({
                    "process_list": process_list_str,
                    "context": context,
                    "issues": issues_str,
                })

            issues = _validate_extraction(result.model_dump(), process_list)
            attempts_log.append({"attempt": attempt, "issues": issues})

            if best_issues is None or len(issues) < len(best_issues):
                best_result, best_issues = result, issues

            if not issues:
                print(f"  Attempt {attempt}: all checks passed.")
                break

            print(f"WARNING: attempt {attempt} violated its own rules:")
            for issue in issues:
                print(f"  - {issue}")

            if attempt < max_attempts:
                print(f"  Retrying (attempt {attempt + 1} of {max_attempts})...")
                issues_str = "\n".join(f"  - {issue}" for issue in issues)

        if best_issues:
            print(f"WARNING: still {len(best_issues)} issue(s) after {len(attempts_log)} "
                  f"attempt(s) — proceeding with the best attempt found:")
            for issue in best_issues:
                print(f"  - {issue}")

        _write_validation_debug({"attempts": attempts_log, "final_issues": best_issues})

        return best_result, best_issues, attempts_log

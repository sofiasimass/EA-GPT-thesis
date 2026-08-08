import os
import asyncio
import logging
from dotenv import load_dotenv
from openai import OpenAI
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
from typing import List

from .bsp import Cluster, EAWeight, EAWeightResult, SystemsAnalysisResult, MarketOption, EAComplianceResult, AsIsToBeResult
from .matrix import MatrixResult #output schema

load_dotenv()


def _validate_extraction(data: dict, process_list: List[str]) -> List[str]:
    """
    Checks the extraction result against the hard rules stated in prompt.txt
    (completeness, entity count, orphan entities, dangling references).
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

    n_entities = len(data.get("entities", []))
    if not (10 <= n_entities <= 20):
        issues.append(f"entity count is {n_entities}, must be between 10 and 20")

    entity_names = {e["name"] for e in data.get("entities", [])}
    touched_entities = {op["entity_name"] for op in data.get("operations", [])}
    orphans = entity_names - touched_entities
    if orphans:
        issues.append(f"{len(orphans)} entit(y/ies) defined but never used in any operation: {sorted(orphans)}")

    op_process_names = {op["process_name"] for op in data.get("operations", [])}
    unknown_op_processes = op_process_names - output_names
    if unknown_op_processes:
        issues.append(f"{len(unknown_op_processes)} operation(s) reference a process not in the output's process list: {sorted(unknown_op_processes)[:10]}")

    op_entity_names = {op["entity_name"] for op in data.get("operations", [])}
    unknown_op_entities = op_entity_names - entity_names
    if unknown_op_entities:
        issues.append(f"{len(unknown_op_entities)} operation(s) reference an entity not in the output's entity list: {sorted(unknown_op_entities)[:10]}")

    processes_with_no_ops = output_names - op_process_names
    if processes_with_no_ops:
        issues.append(f"{len(processes_with_no_ops)} process(es) have zero operations and would vanish from the matrix: {sorted(processes_with_no_ops)[:10]}")

    return issues


class Generator:
    def __init__(self, model_name: str = "gpt-4o"):

        self.llm = ChatOpenAI(model=model_name, temperature=0, request_timeout=120)
        self.structured_output = self.llm.with_structured_output(MatrixResult)

    async def _compute_ea_weights(self, current_clusters: List[Cluster], ea_principles: str, user_constraints: str = ""):
        """
        Analyzes the current BSP result and returns a mapping of
        (first_process, second_process): weight to refine the next iteration.

        Keys are always sorted tuples of the original process names so they
        match the lookup keys built inside bsp.py's _affinity / _extract_blocks.
        """
        # Build a canonical name map: lowercase-stripped → original name.
        # This is the source of truth used to validate names the LLM returns.
        all_process_names: List[str] = []
        for c in current_clusters:
            all_process_names.extend(c.processes)
        canonical: dict[str, str] = {p.strip().lower(): p for p in all_process_names}

        cluster_summary = "\n".join([c.summary() for c in current_clusters])

        # Embed the exact process names in the prompt so the LLM cannot drift.
        exact_names_list = "\n".join(f"  - {p}" for p in all_process_names)

        prompt_path = os.path.join(os.path.dirname(__file__), "..", "resources", "bsp_prompt.txt")
        with open(prompt_path, "r", encoding="utf-8") as f:
            base_instructions = f.read()
        prompt = ChatPromptTemplate.from_template(base_instructions + "\n\n")

        weight_llm = self.llm.with_structured_output(EAWeightResult)

        # Let exceptions propagate — a silent {} hides real problems and
        # makes the final matrix identical to the initial one.
        response = await (prompt | weight_llm).ainvoke({
            "ea_principles": ea_principles,
            "user_constraints": user_constraints if user_constraints.strip() else "(none provided)",
            "cluster_summary": cluster_summary,
            "exact_names_list": exact_names_list,
        })

        weight_map = {}
        reasoning_map = {}
        skipped = []
        for bias in response.biases:
            # Normalize both names to match against canonical map.
            norm_a = bias.first_process.strip().lower()
            norm_b = bias.second_process.strip().lower()

            resolved_a = canonical.get(norm_a)
            resolved_b = canonical.get(norm_b)

            if resolved_a is None or resolved_b is None:
                # LLM hallucinated a name — skip so it doesn't silently zero-out.
                skipped.append((bias.first_process, bias.second_process))
                continue

            # Keys must be sorted so (A,B) == (B,A) everywhere in bsp.py.
            pair = tuple(sorted([resolved_a, resolved_b]))
            weight_map[pair] = bias.weight
            reasoning_map[pair] = bias.reasoning
            print(f"  EA Bias: {pair} -> {bias.weight:+.2f}  ({bias.reasoning})")

        if skipped:
            print(f"  WARNING: {len(skipped)} bias(es) skipped — LLM used unrecognised names: {skipped}")

        if not weight_map:
            print("  WARNING: No valid EA weights were produced. "
                  "The final matrix will be identical to the initial one.")
        else:
            print(f"  {len(weight_map)} EA weight(s) applied successfully.")

        return weight_map, reasoning_map

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
            "DIIEF":  ("1.0", "data storage uniqueness — 1 = no entity redundancy"),
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
            "DIIEF":  ("1.0", "data storage uniqueness — 1 = no entity redundancy"),
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

    async def extract(self, context: str, process_list: list):
        prompt_path = os.path.join(os.path.dirname(__file__), "..", "resources", "prompt.txt")
        with open(prompt_path, "r", encoding="utf-8") as f:
            base_instructions = f.read()

        prompt = ChatPromptTemplate.from_template(
            base_instructions + "\n"
            "TEXT (may be empty — see \"WHEN NO CONTEXT IS PROVIDED\" above):\n{context}\n"
            "Extracted CRUD Matrix:"
        )

        process_list_str = "\n".join(f"  - {p}" for p in process_list)
        result = await (prompt | self.structured_output).ainvoke({
            "process_list": process_list_str,
            "context": context
        })

        issues = _validate_extraction(result.model_dump(), process_list)
        if not issues:
            return result

        print("WARNING: extraction violated its own rules:")
        for issue in issues:
            print(f"  - {issue}")
        print("  Retrying once with a stronger reminder...")

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
        issues_str = "\n".join(f"  - {issue}" for issue in issues)
        retry_result = await (retry_prompt | self.structured_output).ainvoke({
            "process_list": process_list_str,
            "context": context,
            "issues": issues_str,
        })

        remaining_issues = _validate_extraction(retry_result.model_dump(), process_list)
        if remaining_issues:
            print("WARNING: retry still has unresolved issues — proceeding anyway:")
            for issue in remaining_issues:
                print(f"  - {issue}")
        else:
            print("  Retry succeeded — all checks passed.")

        return retry_result

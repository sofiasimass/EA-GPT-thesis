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

    async def extract(self, context: str, process_list: list):
        prompt_path = os.path.join(os.path.dirname(__file__), "..", "resources", "prompt.txt")
        with open(prompt_path, "r", encoding="utf-8") as f:
            base_instructions = f.read()

        prompt = ChatPromptTemplate.from_template(
            base_instructions + "\n"
            "TEXT:\n{context}\n"
            "Extracted CRUD Matrix:"
        )

        process_list_str = "\n".join(f"  - {p}" for p in process_list)
        return await (prompt | self.structured_output).ainvoke({
            "process_list": process_list_str,
            "context": context
        })

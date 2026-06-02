import os
import asyncio
import logging
from dotenv import load_dotenv
from openai import OpenAI
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
from typing import List

from .bsp import Cluster, EAWeight, EAWeightResult
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

        prompt = ChatPromptTemplate.from_template("""
            SYSTEM ROLE:
            You are a Senior Enterprise Architect assigning pair-level weights to correct a process clustering.

            USER CONSTRAINTS (highest priority — apply these first):
            {user_constraints}

            EA PRINCIPLES (baseline — apply only to pairs not already covered by user constraints):
            {ea_principles}

            CURRENT DRAFT CLUSTERS:
            {cluster_summary}

            EXACT PROCESS NAMES — copy verbatim into first_process / second_process:
            {exact_names_list}

            INSTRUCTIONS:

            STEP 1 — Parse USER CONSTRAINTS first.
            For each constraint that names a specific application or system covering multiple domains:
              a) List every process from EXACT PROCESS NAMES that belongs to each domain.
              b) For each cross-domain pair that belongs to the same named application, assign a weight
                 proportional to how tightly the constraint binds them — use the full range 0 to +0.5.
                 Reserve +0.5 only for pairs that are clearly core to the application's purpose.
                 Use smaller values for processes that only loosely relate to the application.
              c) Do NOT assign +0.5 to every pair — differentiate by strength of association.
              d) Do NOT skip cross-cluster pairs. Those are the most important to fix.

            Example: if the user says "AppX manages both suppliers and stock":
              - Supplier processes: [P1, P2]
              - Stock processes:    [P3, P4]
              - Core pairs get higher weights; tangentially related processes get lower weights.

            STEP 2 — Apply EA PRINCIPLES to pairs not covered by STEP 1.
              - Processes clustered together that violate principles → NEGATIVE weight (down to -0.5).
              - Processes split across clusters that should share a domain → POSITIVE weight (up to +0.5).
              - Use the full range: strong violations → closer to -0.5; mild misfits → closer to 0.

            RULES:
            - User constraint pairs ALWAYS take precedence over EA principle pairs.
            - Use the full weight range — do not default everything to +0.5.
            - first_process and second_process MUST be copied EXACTLY from EXACT PROCESS NAMES — no paraphrasing.
        """)

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
            print(f"  EA Bias: {pair} -> {bias.weight:+.2f}  ({bias.reasoning})")

        if skipped:
            print(f"  WARNING: {len(skipped)} bias(es) skipped — LLM used unrecognised names: {skipped}")

        if not weight_map:
            print("  WARNING: No valid EA weights were produced. "
                  "The final matrix will be identical to the initial one.")
        else:
            print(f"  {len(weight_map)} EA weight(s) applied successfully.")

        return weight_map

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
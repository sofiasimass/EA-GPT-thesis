import os
import asyncio
import logging
from dotenv import load_dotenv
from openai import OpenAI
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
from typing import List

from .bsp import Cluster, EAWeight, EAWeightResult, SystemsAnalysisResult, MarketOption
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

            STEP 1 — Parse USER CONSTRAINTS first. There are two types — handle both:

            TYPE A — Application/system constraints (e.g. "AppX manages both suppliers and stock"):
              a) Identify every process from EXACT PROCESS NAMES that belongs to each domain named.
              b) For each cross-domain pair that belongs to the same named application, assign a weight
                 proportional to how tightly the constraint binds them — use the full range 0 to +0.5.
                 Reserve +0.5 only for pairs that are clearly core to the application's purpose.
              c) Do NOT assign +0.5 to every pair — differentiate by strength of association.
              d) Do NOT skip cross-cluster pairs. Those are the most important to fix.

            TYPE B — Operational/transactional constraints (e.g. "operation X and operation Y must reside in the same service"):
              a) Identify the operation(s) mentioned (e.g. "order creation", "payment processing").
              b) From EXACT PROCESS NAMES, list ALL processes that perform each operation — not just the
                 primary owner, but every process that executes that operation in any context
                 (e.g. dine-in order creation, take-away order creation, etc.).
              c) For every pair where one process performs operation X and the other performs operation Y,
                 assign a high positive weight (+0.4 to +0.5). These pairs are the most critical to co-locate.
              d) Do NOT skip cross-cluster pairs — those are exactly the ones the constraint is meant to fix.

            Example TYPE B: if the user says "order creation and payment processing must be in the same service":
              - Order creation processes: [Kitchen Production, Dine-In Service, Premium Take-away]
              - Payment processing processes: [Financial Management]
              - Assign high weights (+0.4 to +0.5) to ALL pairs: (Kitchen Production, Financial Management),
                (Dine-In Service, Financial Management), (Premium Take-away, Financial Management).

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

        prompt = ChatPromptTemplate.from_template("""
You are a Senior Enterprise Architect producing a final assessment of a BSP (Business Systems Planning) clustering exercise.

FINAL CLUSTERS:
{cluster_details}

USER CONSTRAINTS APPLIED ACROSS ITERATIONS (in order):
{user_constraints_history}

EA WEIGHT REASONING (why pairs were boosted or penalised at each iteration):
{weight_reasoning_history}

TASK:
For each cluster above, produce:
1. suggested_name          — a short, business-meaningful system name (e.g. "Customer Relationship Management").
2. description             — 2–3 sentences explaining what this system is responsible for, grounded in its processes, entities, and the architectural decisions made during the exercise.
3. build_or_buy            — one of: "Build", "Buy", or "Hybrid".
   - Buy    : the system maps cleanly onto an existing commercial product category.
   - Build  : the system's logic is too specific or differentiated to be well served by off-the-shelf software.
   - Hybrid : a commercial product covers the core but meaningful customisation or extension is required.
4. build_or_buy_rationale  — 1–2 sentences justifying the decision based on this system's specificity and market coverage.
5. market_options          — up to 3 real, widely-adopted commercial products relevant to this system.
   - Only include products if recommendation is Buy or Hybrid.
   - Leave the list empty if recommendation is Build.
   - Do not invent product names. Use real products only (e.g. Salesforce, SAP S/4HANA, Workday, ServiceNow, Microsoft Dynamics 365, Oracle NetSuite, etc.).

Rules:
- Base naming and description on the cluster content AND the reasoning trail — constraints and weight adjustments reveal design intent that process/entity names alone may not capture.
- Output one SystemDescription per cluster, using the cluster's numeric id as cluster_id.
        """)

        systems_llm = self.llm.with_structured_output(SystemsAnalysisResult)
        return await (prompt | systems_llm).ainvoke({
            "cluster_details": cluster_details,
            "user_constraints_history": user_constraints_history.strip() if user_constraints_history.strip() else "(no iterations — initial clustering accepted)",
            "weight_reasoning_history": weight_reasoning_history.strip() if weight_reasoning_history.strip() else "(no weight adjustments made)",
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

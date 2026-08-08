"""
bsp.py — IBM Business System Planning (BSP) clustering
-------------------------------------------------------
Pure algorithmic implementation of the 4-step BSP method.
No LLM involved. Takes a CRUD matrix and returns clusters.

Techniques used:
  - Weighted Jaccard similarity (process and entity grouping)
  - Greedy nearest-neighbour reordering (Step 3)
  - Union-Find for atomic co-location constraint (Step 4)

BSP Steps:
  Step 1 — Entity ownership: the process that CREATES an entity "owns" it.
            Ties broken by C > U > D > R priority.
  Step 2 — Initial clusters: scaffold based on owned-entity groups.
            Result is discarded; used only for conceptual correctness.
  Step 3 — Affinity reordering: reorder rows/cols so processes and entities
            with the most shared operations end up adjacent, maximising
            block-diagonal density in the matrix.
  Step 4 — Block extraction: group adjacent processes by weighted Jaccard
            similarity, group entities likewise, then enforce constraints:
              • Atomic processes: all entities they WRITE (C/U/D) must land
                in the same cluster (ACID — writes cannot span two systems).
                Entities they only READ may live in other clusters.
              • E2E processes: their entities may be split across clusters;
                the process itself is recorded as spanning those clusters,
                representing cross-system integration dependencies.

process_type values (set by the LLM during extraction):
  "atomic"     — single indivisible transaction; cluster boundaries drawn tightly.
  "end_to_end" — spans multiple departments; cluster boundaries may split it.
  "ambiguous"  — treated as atomic (safe default).
"""

from __future__ import annotations
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Set
import pandas as pd
from pydantic import BaseModel, Field

#schemas

class EAWeight(BaseModel):
    first_process: str = Field(description="The name of the first process in the pair")
    second_process: str = Field(description="The name of the second process in the pair")
    weight: float = Field(description="The bias value between -0.5 and 0.5")
    reasoning: str = Field(description="Architectural justification for this weight")

class EAWeightResult(BaseModel):
    biases: List[EAWeight]


from typing import Literal

class MarketOption(BaseModel):
    name: str = Field(description="Name of the commercial software solution (e.g. Salesforce, SAP S/4HANA, Workday)")
    fit_rationale: str = Field(description="One sentence explaining why this solution fits this system's responsibilities")

class SystemDescription(BaseModel):
    cluster_id: int = Field(description="The numeric ID of the cluster this description refers to")
    suggested_name: str = Field(description="A business-meaningful name for this system (e.g. 'Customer Relationship Management')")
    description: str = Field(description="2-3 sentences describing what this system is responsible for, based on its processes and entities")
    build_or_buy: Literal["Build", "Buy", "Hybrid"] = Field(
        description=(
            "Build — the system's logic is too specific to this business to be covered by off-the-shelf software; "
            "Buy — the system maps well onto an existing commercial product; "
            "Hybrid — a commercial product covers the core but significant customisation or extension is needed"
        )
    )
    build_or_buy_rationale: str = Field(
        description="1-2 sentences justifying the Build/Buy/Hybrid recommendation in terms of this system's specificity and market coverage"
    )
    market_options: List[MarketOption] = Field(
        description="Up to 3 real commercial products relevant to this system. Leave empty if recommendation is Build."
    )

class SystemsAnalysisResult(BaseModel):
    systems: List[SystemDescription]


class PrincipleEvaluation(BaseModel):
    principle: str = Field(description="The EA principle being evaluated, copied verbatim from the principles list")
    status: Literal["compliant", "violated", "partial"] = Field(
        description="compliant — the final clustering fully respects this principle; "
                    "violated — the clustering clearly breaks this principle; "
                    "partial — the clustering partially respects it but with notable exceptions"
    )
    justification: str = Field(description="1-2 sentences grounding the verdict in specific clusters, processes, or entities")

class EAComplianceResult(BaseModel):
    evaluations: List[PrincipleEvaluation]


class TransformationStep(BaseModel):
    step_number: int = Field(description="Sequential order of this migration step, starting at 1")
    action: Literal["merge", "split", "move", "create", "retire"] = Field(
        description=(
            "merge — combine two or more existing applications into one; "
            "split — divide one existing application into two; "
            "move  — relocate specific processes from one application to another; "
            "create — build a new application for processes with no suitable current home; "
            "retire — decommission an application that becomes empty or redundant after migration"
        )
    )
    description: str = Field(description="One concrete sentence describing the action, naming specific processes and applications")
    rationale: str = Field(description="One sentence explaining why this step is required")
    affected_processes: List[str] = Field(description="Names of processes involved in this step")
    affected_applications: List[str] = Field(description="Names of current or proposed applications involved in this step")


class SystemAlignment(BaseModel):
    proposed_cluster_id: int = Field(description="The numeric ID of the proposed cluster")
    proposed_system_name: str = Field(description="The suggested name of the proposed system")
    current_applications: List[str] = Field(
        description="Names of existing applications that contain at least one process belonging to this proposed system"
    )
    processes_to_acquire: List[str] = Field(
        description="Processes currently in other applications that need to move INTO this proposed system"
    )
    processes_to_release: List[str] = Field(
        description="Processes currently grouped here that need to move OUT to a different proposed system"
    )
    alignment_summary: str = Field(
        description="1-2 sentences summarising how well the current state aligns with this proposed system and what the main gap is"
    )


class AsIsToBeResult(BaseModel):
    overall_summary: str = Field(
        description="2-3 sentences describing the overall gap between the current application landscape and the proposed architecture, referencing the metric improvements"
    )
    alignments: List[SystemAlignment] = Field(
        description="One alignment entry per proposed system/cluster"
    )
    transformation_steps: List[TransformationStep] = Field(
        description="Concrete, ordered migration steps to transform the current landscape into the proposed architecture"
    )
    estimated_complexity: Literal["Low", "Medium", "High"] = Field(
        description=(
            "Low — few process moves, minimal application changes; "
            "Medium — moderate restructuring across several applications; "
            "High — significant replatforming, many cross-system migrations, or new systems to build"
        )
    )
    complexity_rationale: str = Field(
        description="1-2 sentences justifying the complexity estimate based on the number and nature of required changes"
    )


@dataclass
class Cluster:
    id: int
    name: str
    processes: List[str] = field(default_factory=list)
    entities: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "processes": self.processes,
            "entities": self.entities,
        }

    def summary(self) -> str:
        procs = ", ".join(self.processes) or "—"
        ents  = ", ".join(self.entities)  or "—"
        return (
            f"  Cluster {self.id}: {self.name}\n"
            f"    Processes : {procs}\n"
            f"    Entities  : {ents}"
        )


@dataclass
class BSPResult:
    clusters: List[Cluster]
    reordered_matrix: pd.DataFrame   # rows=processes, cols=entities
    entity_owners: Dict[str, str]    # entity → owning process
    e2e_span: Dict[str, List[int]] = field(default_factory=dict)  # e2e process → cluster ids it spans


# Numeric weights for CRUD operations.
# C=4 (highest) because creation implies full ownership of the entity.
# Used in affinity calculations and entity ownership resolution.
_OP_PRIORITY = {"C": 4, "U": 3, "D": 2, "R": 1}


def _op_weight(op) -> int:
    """Numeric weight for a CRUD op cell (handles NaN and combined ops like 'CRU')."""
    if op is None or (isinstance(op, float) and pd.isna(op)):
        return 0
    return max((_OP_PRIORITY.get(ch, 0) for ch in str(op).upper()), default=0)

def _affinity(df: pd.DataFrame, a: str, b: str, axis: str, ea_weights: Dict[tuple, float] = None) -> float:
    """
    Measure affinity between two processes (axis='row') or two entities (axis='col').

    For each shared item, sum min(w_a, w_b) — the minimum of the two operation weights.
    This captures "mutual commitment": two processes that both CREATE an entity score
    higher than one that creates and one that only reads.

      C(4) & C(4) → min=4  (both owners — maximum affinity)
      C(4) & U(3) → min=3  (creator + updater — strongly related)
      C(4) & R(1) → min=1  (producer/consumer — weaker link)
      R(1) & R(1) → min=1  (both read-only — coincidence, not co-ownership)

    Result is normalised to [0,1] by dividing by the theoretical maximum (4 × item count),
    so it stays in the same scale as the weighted Jaccard used in _extract_blocks.

    ea_weights (architect-supplied biases in [-0.5, 0.5]) are added directly on top,
    allowing targeted boosts or penalties without a global scaling constant.
    """
    ea_weights = ea_weights or {}
    max_possible = 4 * (len(df.columns) if axis == "row" else len(df.index))

    if axis == "row":
        # Compara dois processos: itera pelas entidades (colunas)
        shared = sum(
            min(_op_weight(df.at[a, col]), _op_weight(df.at[b, col]))
            for col in df.columns
            if _op_weight(df.at[a, col]) > 0 and _op_weight(df.at[b, col]) > 0
        )
    else:
        # Compara duas entidades: itera pelos processos (linhas)
        shared = sum(
            min(_op_weight(df.at[row, a]), _op_weight(df.at[row, b]))
            for row in df.index
            if _op_weight(df.at[row, a]) > 0 and _op_weight(df.at[row, b]) > 0
        )

    shared_normalised = shared / max_possible if max_possible else 0

    # Bias arquitectural: somado directamente — weights em [-0.5, 0.5] têm
    # impacto proporcional sobre uma afinidade entre [0, 1].
    pair = tuple(sorted([a, b]))
    weight = ea_weights.get(pair, 0)

    return shared_normalised + weight

def _greedy_reorder(items: List[str], affinity_fn) -> List[str]:
    """
    Greedy nearest-neighbour reordering.
    Start with the highest-total-affinity item, then always pick the
    unvisited neighbour with the highest affinity to the last placed item.
    """
    if len(items) <= 1:
        return items

    totals = {i: sum(affinity_fn(i, j) for j in items if j != i) for i in items}
    seed   = max(totals, key=totals.get)

    ordered   = [seed]
    remaining = set(items) - {seed}

    while remaining:
        last = ordered[-1]
        nxt  = max(remaining, key=lambda x: affinity_fn(last, x))
        ordered.append(nxt)
        remaining.remove(nxt)

    return ordered


def _cluster_name(entities: List[str]) -> str:
    if not entities:
        return "Unnamed"
    return entities[0] if len(entities) == 1 else " / ".join(entities[:2])


def _compute_entity_owners(df: pd.DataFrame) -> Dict[str, str]:
    """
    For each entity (column), the process (row) with the highest-priority
    CRUD operation owns it.  C > U > R > D.  Ties broken by first occurrence.
    """
    owners: Dict[str, str] = {}
    for entity in df.columns:
        best_proc, best_w = None, -1
        for proc in df.index:
            w = _op_weight(df.at[proc, entity])
            if w > best_w:
                best_w, best_proc = w, proc
        if best_proc:
            owners[entity] = best_proc
    return owners



def _reorder_matrix(df: pd.DataFrame, ea_weights: Dict[tuple, float] = None) -> pd.DataFrame:
    """
    Reorder rows (processes) and columns (entities) so that items with high
    mutual affinity end up adjacent, producing a block-diagonal structure.
    ea_weights shift affinity scores before reordering, so architect-supplied
    biases influence the final layout directly.
    """
    new_procs = _greedy_reorder(
        list(df.index), 
        lambda a, b: _affinity(df, a, b, "row", ea_weights)
    )
    new_ents  = _greedy_reorder(
        list(df.columns), 
        lambda a, b: _affinity(df, a, b, "col", ea_weights)
    )
    return df.loc[new_procs, new_ents]

def _extract_blocks(df: pd.DataFrame, density_threshold: float = 0.5, ea_weights: Dict[tuple, float] = None, process_types: Dict[str, str] = None) -> List[Cluster]:

    ea_weights   = ea_weights or {}
    process_types = process_types or {}
    procs    = list(df.index)
    entities = list(df.columns)

    def _weighted_jaccard_procs(p1: str, p2: str) -> float:
        """Jaccard ponderado entre dois processos, iterando pelas entidades."""
        intersection = sum(
            min(_op_weight(df.at[p1, e]), _op_weight(df.at[p2, e]))
            for e in entities
        )
        union_sum = sum(
            max(_op_weight(df.at[p1, e]), _op_weight(df.at[p2, e]))
            for e in entities
        )
        return intersection / union_sum if union_sum else 0

    def _weighted_jaccard_ents(e1: str, e2: str) -> float:
        """Jaccard ponderado entre duas entidades, iterando pelos processos."""
        intersection = sum(
            min(_op_weight(df.at[p, e1]), _op_weight(df.at[p, e2]))
            for p in procs
        )
        union_sum = sum(
            max(_op_weight(df.at[p, e1]), _op_weight(df.at[p, e2]))
            for p in procs
        )
        return intersection / union_sum if union_sum else 0

    # --- Step 4a: Group processes ---
    # Compare each process to the *average* similarity of the whole current group,
    # not just the last element. This avoids chaining artefacts where two dissimilar
    # processes end up in the same group because they each happen to be similar to
    # the process immediately before them.
    proc_groups: List[List[str]] = [[procs[0]]]
    for i in range(1, len(procs)):
        curr = procs[i]
        current_group = proc_groups[-1]

        group_similarities = []
        for member in current_group:
            jaccard = _weighted_jaccard_procs(curr, member)
            pair = tuple(sorted([member, curr]))
            weight = ea_weights.get(pair, 0)  # architect bias shifts the threshold
            group_similarities.append(jaccard + weight)

        avg_similarity = sum(group_similarities) / len(group_similarities)

        print(f"Average similarity of {curr} to group {current_group} is {avg_similarity:.2f}")

        if avg_similarity >= density_threshold:
            print("JOINED TO CURRENT GROUP")
            proc_groups[-1].append(curr)
        else:
            print("STARTED NEW GROUP")
            proc_groups.append([curr])

    # --- Step 4b: Group entities ---
    # ea_weights are process-pair biases and do not apply here.
    # Entity grouping uses only the structural Jaccard signal.
    ent_groups: List[List[str]] = [[entities[0]]]
    for j in range(1, len(entities)):
        prev = ent_groups[-1][-1]
        curr = entities[j]
        jaccard = _weighted_jaccard_ents(prev, curr)
        if jaccard >= density_threshold:
            ent_groups[-1].append(curr)
        else:
            ent_groups.append([curr])

    # --- Co-location constraint: atomic processes force their entities into the same group ---
    # Uses union-find so cascading merges (A+B then B+C) are handled correctly.
    if process_types:
        parent = {e: e for e in entities}

        def _find(e: str) -> str:
            while parent[e] != e:
                parent[e] = parent[parent[e]]
                e = parent[e]
            return e

        def _union(e1: str, e2: str) -> None:
            r1, r2 = _find(e1), _find(e2)
            if r1 != r2:
                parent[r2] = r1

        # Preserve existing Jaccard groupings
        for eg in ent_groups:
            for i in range(1, len(eg)):
                _union(eg[0], eg[i])

        # Force co-location for atomic processes — writes only (C/U/D).
        # Reads don't need ACID guarantees and may cross cluster boundaries.
        for proc in procs:
            if process_types.get(proc, "atomic") == "atomic":
                touched = [e for e in entities if _op_weight(df.at[proc, e]) > _OP_PRIORITY["R"]]
                for i in range(1, len(touched)):
                    _union(touched[0], touched[i])

        # Rebuild ent_groups from union-find results
        groups_dict: Dict[str, List[str]] = defaultdict(list)
        for e in entities:
            groups_dict[_find(e)].append(e)

        entity_order = {e: i for i, e in enumerate(entities)}
        ent_groups = [sorted(g, key=lambda e: entity_order[e]) for g in groups_dict.values()]
        ent_groups.sort(key=lambda g: entity_order[g[0]])

    # --- Pair process groups ↔ entity groups: score = soma dos op_weights (já não binário) ---
    clusters: List[Cluster] = []
    for cid, pg in enumerate(proc_groups, start=1):
        best_eg, best_score = ent_groups[0], -1
        for eg in ent_groups:
            score = sum(_op_weight(df.at[p, e]) for p in pg for e in eg)
            if score > best_score:
                best_score, best_eg = score, eg

        existing = next((c for c in clusters if set(c.entities) == set(best_eg)), None)
        if existing:
            existing.processes.extend(pg)
        else:
            clusters.append(Cluster(
                id=cid,
                name=_cluster_name(best_eg),
                processes=list(pg),
                entities=list(best_eg),
            ))

    # --- Assign any unclaimed entity groups ---
    claimed = {e for c in clusters for e in c.entities}
    for eg in ent_groups:
        if any(e not in claimed for e in eg):
            best_c, best_score = clusters[0], -1
            for c in clusters:
                score = sum(_op_weight(df.at[p, e]) for p in c.processes for e in eg)
                if score > best_score:
                    best_score, best_c = score, c
            if best_score > 0:
                new_entities = [e for e in eg if e not in claimed]
                best_c.entities.extend(new_entities)
                best_c.name = _cluster_name(best_c.entities)
                claimed.update(new_entities)

    return clusters

def _absorb_pure_readers(clusters: List[Cluster], df: pd.DataFrame, process_types: Dict[str, str] = None) -> List[Cluster]:
    """
    Any atomic process with zero C/U operations is a pure consumer.
    Move it into whichever cluster owns (C) the most entities it reads.
    E2E processes are skipped — they legitimately read across cluster boundaries.
    """
    process_types = process_types or {}
    owner_cluster = {}  # entity → cluster index
    for i, c in enumerate(clusters):
        for proc in c.processes:
            for ent in df.columns:
                if _op_weight(df.at[proc, ent]) == _OP_PRIORITY["C"]:
                    owner_cluster[ent] = i

    for c in clusters:
        readers_to_move = []
        for proc in c.processes:
            if process_types.get(proc) == "end_to_end":
                continue
            ops = [df.at[proc, e] for e in df.columns if _op_weight(df.at[proc, e]) > 0]
            is_pure_reader = all(str(op).upper() in ("R", "nan") for op in ops)
            if is_pure_reader:
                readers_to_move.append(proc)

        for proc in readers_to_move:
            read_ents = [e for e in df.columns if str(df.at[proc, e]).upper() == "R"]
            best_cluster, best_score = c, 0
            for ent in read_ents:
                ci = owner_cluster.get(ent)
                if ci is not None and ci != clusters.index(c):
                    score = sum(1 for e in read_ents if owner_cluster.get(e) == ci)
                    if score > best_score:
                        best_score, best_cluster = score, clusters[ci]
            if best_cluster is not c:
                c.processes.remove(proc)
                best_cluster.processes.append(proc)

    # Remove any now-empty clusters, then renumber so IDs stay contiguous 1..N —
    # otherwise a dropped cluster leaves a gap (e.g. 1,2,3,5,6…) that every
    # downstream LLM call and the UI would otherwise have to explain away.
    survivors = [c for c in clusters if c.processes]
    for new_id, c in enumerate(survivors, start=1):
        c.id = new_id
    return survivors

def _compute_e2e_span(df: pd.DataFrame, clusters: List[Cluster], process_types: Dict[str, str]) -> Dict[str, List[int]]:
    """For each E2E process, return the list of cluster IDs whose entities it touches."""
    span: Dict[str, List[int]] = {}
    for proc in df.index:
        if process_types.get(proc) != "end_to_end":
            continue
        touched_ids = [
            c.id for c in clusters
            if any(_op_weight(df.at[proc, e]) > 0 for e in c.entities if e in df.columns)
        ]
        span[proc] = touched_ids
    return span


def compute_isa_metrics(
    clusters: List[Cluster],
    crud_matrix: Dict[str, Dict[str, str]],
    process_types: Dict[str, str],
) -> Dict[str, float]:
    """
    Computes ISA quality metrics (Vasconcelos et al.) for a given clustering.
    All metrics in [0, 1]; higher is better.

    RSF   : Average IS blocks per process — 1 = every process lives in exactly one system
    NAIEF : Entity write responsibility — 1 = single source of truth per entity (CUD in one cluster)
    LCOISF: Cluster cohesion — 1 = every entity pair in a cluster shares at least one common process
    CPSMF : Critical/non-critical isolation — 1 = atomic and E2E processes never share a cluster
    DIIEF : Data access uniqueness — 1 = each entity is touched by processes in only one cluster
    """
    if not crud_matrix or not clusters:
        return {"RSF": 0.0, "NAIEF": 0.0, "LCOISF": 0.0, "CPSMF": 0.0, "DIIEF": 0.0}

    df = pd.DataFrame(crud_matrix).T
    all_procs = list(df.index)
    all_ents  = list(df.columns)

    proc_cluster_ids: Dict[str, Set[int]] = {p: set() for p in all_procs}
    ent_cud_clusters: Dict[str, Set[int]] = {e: set() for e in all_ents}
    ent_any_clusters: Dict[str, Set[int]] = {e: set() for e in all_ents}

    for c in clusters:
        for p in c.processes:
            if p in proc_cluster_ids:
                proc_cluster_ids[p].add(c.id)
            if p not in df.index:
                continue
            for e in all_ents:
                if e not in df.columns:
                    continue
                w = _op_weight(df.at[p, e])
                if w > _OP_PRIORITY["R"]:
                    ent_cud_clusters[e].add(c.id)
                if w > 0:
                    ent_any_clusters[e].add(c.id)

    # RSF: total processes / sum of cluster-spans per process (ideal = 1)
    total_spans = sum(max(1, len(v)) for v in proc_cluster_ids.values())
    rsf = len(all_procs) / total_spans if total_spans else 1.0

    # NAIEF: total entities / sum of CUD-cluster-spans per entity (ideal = 1)
    total_cud = sum(max(1, len(v)) for v in ent_cud_clusters.values())
    naief = len(all_ents) / total_cud if total_cud else 1.0

    # LCOISF: for each cluster, fraction of entity pairs that share at least one process
    cohesion_vals = []
    for c in clusters:
        procs_in = [p for p in c.processes if p in df.index]
        ents_in  = [e for e in c.entities  if e in df.columns]
        if len(ents_in) < 2:
            cohesion_vals.append(1.0)
            continue
        connected = total_ep = 0
        for i_e, e1 in enumerate(ents_in):
            for e2 in ents_in[i_e + 1:]:
                total_ep += 1
                if any(
                    _op_weight(df.at[p, e1]) > 0 and _op_weight(df.at[p, e2]) > 0
                    for p in procs_in
                ):
                    connected += 1
        cohesion_vals.append(connected / total_ep if total_ep else 1.0)
    lcoisf = sum(cohesion_vals) / len(cohesion_vals) if cohesion_vals else 1.0

    # CPSMF: isolation of atomic (critical) vs end_to_end (non-critical) processes
    def _is_critical(p: str) -> bool:
        return process_types.get(p, "atomic") in ("atomic", "ambiguous")

    def _is_critical_cluster(c: Cluster) -> bool:
        if not c.processes:
            return True
        return sum(1 for p in c.processes if _is_critical(p)) >= len(c.processes) / 2

    mismatch = sum(
        1 for c in clusters
        for p in c.processes
        if _is_critical(p) != _is_critical_cluster(c)
    )
    cpsmf = 1.0 - (mismatch / len(all_procs)) if all_procs else 1.0

    # DIIEF: total entities / sum of any-operation-cluster-spans per entity (ideal = 1)
    total_any = sum(max(1, len(v)) for v in ent_any_clusters.values())
    diief = len(all_ents) / total_any if total_any else 1.0

    return {
        "RSF":    round(rsf,    3),
        "NAIEF":  round(naief,  3),
        "LCOISF": round(lcoisf, 3),
        "CPSMF":  round(cpsmf,  3),
        "DIIEF":  round(diief,  3),
    }


def as_is_to_clusters(
    as_is_mapping: Dict[str, List[str]],
    crud_matrix: Dict[str, Dict[str, str]],
) -> List[Cluster]:
    """
    Converts an As-Is application mapping { app_name: [process, ...] } into
    Cluster objects for metric computation.  Entities are inferred from the
    CRUD matrix — every entity touched (any operation) by a cluster's processes.
    """
    df = pd.DataFrame(crud_matrix).T
    clusters = []
    for cid, (app, procs) in enumerate(as_is_mapping.items(), start=1):
        valid = [p for p in procs if p in df.index]
        ents  = {
            e for p in valid for e in df.columns
            if _op_weight(df.at[p, e]) > 0
        }
        clusters.append(Cluster(id=cid, name=app, processes=valid, entities=list(ents)))
    return clusters


def matrix_to_clusters(
    matrix_data: Dict[str, Dict[str, str]],
    process_system: Dict[str, str],
) -> List[Cluster]:
    """
    Groups a self-contained CRUD matrix into Cluster objects using an explicit
    process -> system/application mapping, instead of computing clusters via BSP.
    Used for an independently-modeled As-Is landscape: the matrix and the system
    grouping both come from the same source (the user's own CSV), so — unlike
    as_is_to_clusters — nothing here is filtered against a *different* matrix's
    process list, and no process/entity is ever silently dropped.
    """
    df = pd.DataFrame(matrix_data).T
    systems: Dict[str, List[str]] = defaultdict(list)
    for proc, system in process_system.items():
        systems[system].append(proc)

    clusters = []
    for cid, (system, procs) in enumerate(systems.items(), start=1):
        ents = {e for p in procs for e in df.columns if _op_weight(df.at[p, e]) > 0}
        clusters.append(Cluster(id=cid, name=system, processes=list(procs), entities=list(ents)))
    return clusters


def run_bsp(matrix_data: Dict[str, Dict[str, str]], density_threshold: float = 0.5, ea_weights: Dict[tuple, float] = None, process_types: Dict[str, str] = None) -> BSPResult:
    """
    Run the full 4-step BSP algorithm.

    Parameters
    ----------
    matrix_data : { process_name: { entity_name: "C"|"R"|"U"|"D" } }
        Raw CRUD matrix dict as stored in Matrix.matrix.

    Returns
    -------
    BSPResult with clusters, reordered_matrix, entity_owners.
    """
    if not matrix_data:
        raise ValueError("CRUD matrix is empty — run the extraction step first.")

    process_types = process_types or {}

    df = pd.DataFrame(matrix_data).T   # rows=processes, cols=entities

    owners    = _compute_entity_owners(df)                                           # Step 1
    reordered = _reorder_matrix(df, ea_weights=ea_weights)                           # Step 2
    clusters  = _extract_blocks(reordered, density_threshold=density_threshold,
                                ea_weights=ea_weights, process_types=process_types)  # Step 4
    final     = _absorb_pure_readers(clusters=clusters, df=df,
                                     process_types=process_types)
    e2e_span  = _compute_e2e_span(df, final, process_types)

    return BSPResult(clusters=final, reordered_matrix=reordered,
                     entity_owners=owners, e2e_span=e2e_span)

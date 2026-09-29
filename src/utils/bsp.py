"""
bsp.py — BSP (IBM Business System Planning) clustering. No LLM calls.

Takes a CRUD matrix { process: { entity: "C"|"R"|"U"|"D" } } and groups
processes and entities into candidate systems (clusters).

Pipeline (see run_bsp):
  1. Reorder rows/columns by affinity so related items sit together (display).
  2. Group entities by weighted Jaccard similarity (average linkage).
  3. Force every entity an atomic process WRITES into the same group (ACID).
  4. Each entity group becomes a cluster; a process is a member of every
     cluster it writes to (C/U/D). Pure readers belong to no cluster.

Architect weights (from the LLM, in [-0.5, 0.5]):
  - entity_weights  : added to entity-pair similarity in step 2, so they are
                      the ONLY weights that change the clusters.
  - process_weights : only change the row order of the displayed matrix and
                      are used to detect conflicts with atomicity. They never
                      change cluster membership.
Neither can silently break atomicity: a conflict is returned as a
PendingConflict and only applied once the architect confirms it.

process_type: "atomic" (one transaction, one system), "end_to_end" (may span
systems) or "ambiguous" (treated as atomic).
"""

from __future__ import annotations
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Literal, Set
import statistics
import pandas as pd
from pydantic import BaseModel, Field

# ---- LLM output schemas (field descriptions are part of the prompt) ----

class ProcessWeight(BaseModel):
    first_process: str = Field(description="The name of the first process in the pair")
    second_process: str = Field(description="The name of the second process in the pair")
    weight: float = Field(description="The bias value between -0.5 and 0.5")
    reasoning: str = Field(description="Architectural justification for this weight")

class EntityWeight(BaseModel):
    first_entity: str = Field(description="The name of the first entity in the pair")
    second_entity: str = Field(description="The name of the second entity in the pair")
    weight: float = Field(description="The bias value between -0.5 and 0.5")
    reasoning: str = Field(description="Architectural justification for this weight")

class ConstraintWeightsResult(BaseModel):
    process_biases: List[ProcessWeight] = Field(default_factory=list)
    entity_biases: List[EntityWeight] = Field(default_factory=list)


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


# ---- clustering data structures ----

# A candidate system: a group of entities plus the processes that write to them
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

    # Text summary of the cluster, used in LLM prompts
    def summary(self) -> str:
        procs = ", ".join(self.processes) or "—"
        ents  = ", ".join(self.entities)  or "—"
        return (
            f"  Cluster {self.id}: {self.name}\n"
            f"    Processes : {procs}\n"
            f"    Entities  : {ents}"
        )


# Everything run_bsp returns
@dataclass
class BSPResult:
    clusters: List[Cluster]
    reordered_matrix: pd.DataFrame   # rows=processes, cols=entities
    entity_owners: Dict[str, str]    # entity → owning process
    process_span: Dict[str, List[int]] = field(default_factory=dict)  # process → cluster ids it writes to (e2e or confirmed override)
    hub_flags: List[HubFlag] = field(default_factory=list)                 # processes/entities that dominate the matrix (warning only)
    pending_conflicts: List[PendingConflict] = field(default_factory=list)  # atomicity vs. weight conflicts awaiting confirmation
    unallocated_reads: Dict[str, List[int]] = field(default_factory=dict)  # process with no cluster → cluster ids it reads (report only)


# A process or entity touching more than half of the other axis (Bureš et al., 2019)
@dataclass
class HubFlag:
    subject: str         # process or entity name
    axis: str            # "process" or "entity"
    coverage: float      # fraction of the other axis this item touches, 0 to 1
    threshold: float


# One merge/split decision for a process-entity pair, with its reason
@dataclass
class Decision:
    subject: str         # process
    entity: str          # entity involved in this decision
    action: str          # "exempt_confirmed" | "force_merge" | "ordinary_merge" | "ordinary_split"
    reasoning: str
    score: float
    threshold: float


# Inputs that decide() needs to judge one entity
@dataclass
class DecisionContext:
    threshold: float
    process_types: Dict[str, str]
    confirmed_overrides: Dict[str, Set[str]]
    similarity_fn: Callable[[str, str], float]
    anchor: str                          # the first entity the process writes; others are compared to it
    ignore_atomicity: bool = False       # True only when testing "what if this process were not atomic?"


# An atomic process forced to keep an entity that the architect's weights want separated
@dataclass
class PendingConflict:
    process: str
    conflicting_process: str
    entity: str
    adjusted_score: float
    threshold: float
    reasoning: str


# One process or entity that changed cluster between two iterations
@dataclass
class ClusterChange:
    subject: str
    from_cluster: int | None
    to_cluster: int | None
    change_type: str     # "atomicity_override" or "ordinary"
    reasoning: str


# Numeric weight of each CRUD operation. C is highest because creating an entity implies owning it.
_OP_PRIORITY = {"C": 4, "U": 3, "D": 2, "R": 1}


# Weight of a matrix cell; a combined cell like "CRU" takes its strongest operation, empty = 0
def _op_weight(op) -> int:
    if op is None or (isinstance(op, float) and pd.isna(op)):
        return 0
    return max((_OP_PRIORITY.get(ch, 0) for ch in str(op).upper()), default=0)


# Owner of each entity = the process with the strongest operation on it (report only)
def _compute_entity_owners(df: pd.DataFrame) -> Dict[str, str]:
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


# ---- hub detection ----

# Flags processes/entities that touch more than `threshold` of the other axis (warning only)
def _detect_hubs(df: pd.DataFrame, threshold: float = 0.5) -> List[HubFlag]:
    flags: List[HubFlag] = []
    n_entities = len(df.columns)
    n_procs = len(df.index)

    for proc in df.index:
        touched = sum(1 for e in df.columns if _op_weight(df.at[proc, e]) > 0)
        coverage = touched / n_entities if n_entities else 0
        if coverage > threshold:
            flags.append(HubFlag(subject=proc, axis="process", coverage=coverage, threshold=threshold))

    for ent in df.columns:
        touched = sum(1 for p in df.index if _op_weight(df.at[p, ent]) > 0)
        coverage = touched / n_procs if n_procs else 0
        if coverage > threshold:
            flags.append(HubFlag(subject=ent, axis="entity", coverage=coverage, threshold=threshold))

    return flags


# Affinity between two processes (axis="row") or two entities (axis="col"), used only to reorder the matrix
def _affinity(df: pd.DataFrame, a: str, b: str, axis: str, process_weights: Dict[tuple, float] = None) -> float:
    process_weights = process_weights or {}
    max_possible = 4 * (len(df.columns) if axis == "row" else len(df.index))

    # Sum min(weight_a, weight_b) over items both touch: two creators (C,C) score 4, creator + reader (C,R) only 1
    if axis == "row":
        shared = sum(
            min(_op_weight(df.at[a, col]), _op_weight(df.at[b, col]))
            for col in df.columns
            if _op_weight(df.at[a, col]) > 0 and _op_weight(df.at[b, col]) > 0
        )
    else:
        shared = sum(
            min(_op_weight(df.at[row, a]), _op_weight(df.at[row, b]))
            for row in df.index
            if _op_weight(df.at[row, a]) > 0 and _op_weight(df.at[row, b]) > 0
        )

    # Normalise by the maximum possible score (4 per item) so it stays in [0, 1]
    shared_normalised = shared / max_possible if max_possible else 0

    # process_weights are keyed by process pairs, so they only affect row (process) order
    pair = tuple(sorted([a, b]))
    weight = process_weights.get(pair, 0)

    return shared_normalised + weight

# Greedy nearest-neighbour ordering: start from the most connected item, then always add the closest one
def _greedy_reorder(items: List[str], affinity_fn) -> List[str]:
    if len(items) <= 1:
        return items

    totals = {i: sum(affinity_fn(i, j) for j in items if j != i) for i in items}
    seed   = max(totals, key=totals.get)

    ordered   = [seed]
    # A list, not a set: set order changes between runs and would make ties non-deterministic
    remaining = [i for i in items if i != seed]

    while remaining:
        last = ordered[-1]
        nxt  = max(remaining, key=lambda x: affinity_fn(last, x))
        ordered.append(nxt)
        remaining.remove(nxt)

    return ordered


# Default cluster name from its first one or two entities
def _cluster_name(entities: List[str]) -> str:
    if not entities:
        return "Unnamed"
    return entities[0] if len(entities) == 1 else " / ".join(entities[:2])


# Reorders rows and columns so related items are adjacent, giving a block-diagonal matrix
def _reorder_matrix(df: pd.DataFrame, process_weights: Dict[tuple, float] = None) -> pd.DataFrame:
    new_procs = _greedy_reorder(
        list(df.index),
        lambda a, b: _affinity(df, a, b, "row", process_weights)
    )
    new_ents  = _greedy_reorder(
        list(df.columns),
        lambda a, b: _affinity(df, a, b, "col", process_weights)
    )
    return df.loc[new_procs, new_ents]


# ---- clustering pipeline ----

# Weighted Jaccard similarity between two processes (axis="row") or two entities (axis="col")
def _weighted_jaccard(df: pd.DataFrame, a: str, b: str, axis: str) -> float:
    # Compare the two rows/columns cell by cell using operation weights (C=4, U=3, D=2, R=1):
    #   similarity = sum(min(w_a, w_b)) / sum(max(w_a, w_b))
    # 1.0 = identical CRUD footprint, 0.0 = nothing in common.
    # e.g. entities X and Y, process P does C on X and U on Y: min=3, max=4 on that row.
    # Used on entities to form clusters, and on processes for the adaptive threshold.
    if axis == "row":
        others = df.columns
        intersection = sum(min(_op_weight(df.at[a, o]), _op_weight(df.at[b, o])) for o in others)
        union_sum    = sum(max(_op_weight(df.at[a, o]), _op_weight(df.at[b, o])) for o in others)
    else:
        others = df.index
        intersection = sum(min(_op_weight(df.at[o, a]), _op_weight(df.at[o, b])) for o in others)
        union_sum    = sum(max(_op_weight(df.at[o, a]), _op_weight(df.at[o, b])) for o in others)

    return intersection / union_sum if union_sum else 0


# Entity similarity function = weighted Jaccard + the architect's entity weight for that pair
def _entity_similarity_with_bias(df: pd.DataFrame, entity_weights: Dict[tuple, float] = None) -> Callable[[str, str], float]:
    entity_weights = entity_weights or {}

    def fn(a: str, b: str) -> float:
        return _weighted_jaccard(df, a, b, "col") + entity_weights.get(tuple(sorted([a, b])), 0)

    return fn


# Average-linkage grouping: each item joins the existing group with the highest AVERAGE similarity, if >= threshold
def _average_linkage_group(
    items: List[str],
    similarity_fn: Callable[[str, str], float],
    threshold: float,
    bias_fn: Callable[[str, str], float] = None,
) -> List[List[str]]:
    # Comparing to the group average (not one member) avoids chaining A~B~C when A and C are unrelated.
    # Every item is compared to ALL groups, not just the last one, so matrix order can't hide a strong match.
    if not items:
        return []

    groups: List[List[str]] = [[items[0]]]
    for curr in items[1:]:
        best_group, best_avg = None, -1.0
        for group in groups:
            similarities = []
            for member in group:
                score = similarity_fn(curr, member)
                if bias_fn:
                    score += bias_fn(curr, member)
                similarities.append(score)
            avg = sum(similarities) / len(similarities)
            if avg > best_avg:
                best_avg, best_group = avg, group

        if best_avg >= threshold:
            best_group.append(curr)
        else:
            groups.append([curr])

    return groups


# Groups entities into clusters: weighted Jaccard between entities + entity_weights as bias
def _group_entities(df: pd.DataFrame, threshold: float, entity_weights: Dict[tuple, float] = None) -> List[List[str]]:
    # This is the step that defines cluster shape, and entity_weights are the only architect weights applied here
    entity_weights = entity_weights or {}

    def bias(a: str, b: str) -> float:
        return entity_weights.get(tuple(sorted([a, b])), 0)

    return _average_linkage_group(
        list(df.columns),
        lambda a, b: _weighted_jaccard(df, a, b, "col"),
        threshold,
        bias_fn=bias,
    )


# All pairwise similarity scores of a list, used to compute the adaptive threshold
def _pairwise_scores(items: List[str], similarity_fn: Callable[[str, str], float]) -> List[float]:
    scores = []
    for i, a in enumerate(items):
        for b in items[i + 1:]:
            scores.append(similarity_fn(a, b))
    return scores


# Adaptive threshold = mean + k × std of this matrix's similarity scores (falls back to 0.5 with too few pairs)
def derive_threshold(
    scores: List[float],
    k: float = 1.0,
    min_pairs: int = 5,
    fallback: float = 0.5,
) -> float:
    # Optional alternative to the fixed 0.5 of Lee (1999), who notes "there is no guideline for
    # determining suitable k". This statistic is this project's own, not taken from the literature.
    # Off by default (run_bsp(adaptive_threshold=False)).
    if len(scores) < min_pairs:
        return fallback
    mean = statistics.fmean(scores)
    stdev = statistics.pstdev(scores)
    return mean + k * stdev


# Union-find over entities: merges groups that must stay together, including chained merges (A+B, B+C)
class _UnionFind:
    def __init__(self, items: List[str], seed_groups: List[List[str]] = None):
        self.parent = {item: item for item in items}
        if seed_groups:
            for group in seed_groups:
                for i in range(1, len(group)):
                    self.union(group[0], group[i])

    def find(self, item: str) -> str:
        while self.parent[item] != item:
            self.parent[item] = self.parent[self.parent[item]]
            item = self.parent[item]
        return item

    def union(self, a: str, b: str) -> None:
        root_a, root_b = self.find(a), self.find(b)
        if root_a != root_b:
            self.parent[root_b] = root_a

    # Final groups, ordered by where their first entity appears in `order`
    def groups(self, order: List[str]) -> List[List[str]]:
        buckets: Dict[str, List[str]] = defaultdict(list)
        for item in order:
            buckets[self.find(item)].append(item)

        order_index = {item: i for i, item in enumerate(order)}
        result = list(buckets.values())
        result.sort(key=lambda g: order_index[g[0]])
        return result


# Decision table (Grad, 1962) for one process-entity pair, rules in priority order:
#   1. architect confirmed an override  -> entity may leave the process's cluster
#   2. process is atomic/ambiguous      -> force the entity into the same cluster (ACID)
#   3. otherwise                        -> merge only if similarity >= threshold
def decide(process: str, entity: str, ctx: DecisionContext) -> Decision:
    if entity in ctx.confirmed_overrides.get(process, set()):
        return Decision(
            subject=process, entity=entity, action="exempt_confirmed",
            reasoning=f"the architect confirmed that {entity} may leave {process}'s cluster",
            score=0.0, threshold=ctx.threshold,
        )

    if not ctx.ignore_atomicity and ctx.process_types.get(process, "atomic") in ("atomic", "ambiguous"):
        return Decision(
            subject=process, entity=entity, action="force_merge",
            reasoning=f"{process} is atomic and writes to {entity}, so they must stay in the same system",
            score=0.0, threshold=ctx.threshold,
        )

    score = ctx.similarity_fn(ctx.anchor, entity)
    if score >= ctx.threshold:
        action = "ordinary_merge"
        reasoning = f"similarity {score:.2f} >= threshold {ctx.threshold:.2f}"
    else:
        action = "ordinary_split"
        reasoning = f"similarity {score:.2f} < threshold {ctx.threshold:.2f}"

    return Decision(subject=process, entity=entity, action=action, reasoning=reasoning,
                     score=score, threshold=ctx.threshold)


# Atomicity rule: all entities an atomic process writes (C/U/D) are merged into one group
def _enforce_atomicity(
    df: pd.DataFrame,
    entities: List[str],
    ent_groups: List[List[str]],
    process_types: Dict[str, str],
    confirmed_overrides: Dict[str, Set[str]],
    threshold: float,
) -> tuple[List[List[str]], List[Decision]]:
    confirmed_overrides = confirmed_overrides or {}
    uf = _UnionFind(entities, seed_groups=ent_groups)
    decisions: List[Decision] = []

    for proc in df.index:
        if process_types.get(proc, "atomic") not in ("atomic", "ambiguous"):
            continue

        # Entities this process writes (anything stronger than a read)
        touched = [e for e in entities if _op_weight(df.at[proc, e]) > _OP_PRIORITY["R"]]
        if not touched:
            continue
        anchor = touched[0]

        ctx = DecisionContext(
            threshold=threshold, process_types=process_types,
            confirmed_overrides=confirmed_overrides,
            similarity_fn=lambda a, b: _weighted_jaccard(df, a, b, "col"),
            anchor=anchor,
        )

        for entity in touched[1:]:
            decision = decide(proc, entity, ctx)
            decisions.append(decision)
            if decision.action == "force_merge":
                uf.union(anchor, entity)
            # "exempt_confirmed" merges nothing: the entity stays where grouping put it

    new_groups = uf.groups(entities)
    return new_groups, decisions


# Finds forced merges that the architect's weights would have split, returned for confirmation
def _detect_pending_conflicts(
    df: pd.DataFrame,
    decisions: List[Decision],
    entity_owners: Dict[str, str],
    process_weights: Dict[tuple, float],
    process_reasoning: Dict[tuple, str],
    threshold: float,
    entity_weights: Dict[tuple, float] = None,
    entity_reasoning: Dict[tuple, str] = None,
) -> List[PendingConflict]:
    # For each forced merge: "if this process were not atomic, would the weighted similarity split them?"
    # The negative signal can come from an entity weight (anchor ↔ entity) and/or a process weight
    # (process ↔ the entity's owner); both are summed.
    process_weights = process_weights or {}
    entity_weights = entity_weights or {}
    if not process_weights and not entity_weights:
        return []

    process_reasoning = process_reasoning or {}
    entity_reasoning = entity_reasoning or {}
    conflicts: List[PendingConflict] = []

    for decision in decisions:
        if decision.action != "force_merge":
            continue

        process = decision.subject
        entity = decision.entity

        touched = [e for e in df.columns if _op_weight(df.at[process, e]) > _OP_PRIORITY["R"]]
        if not touched:
            continue
        anchor = touched[0]

        entity_pair = tuple(sorted([anchor, entity]))
        entity_bias = entity_weights.get(entity_pair, 0)

        # A process weight only counts if the entity is owned by ANOTHER process
        owner = entity_owners.get(entity, "")
        process_pair = tuple(sorted([process, owner]))
        process_bias = process_weights.get(process_pair, 0) if owner and owner != process else 0

        total_bias = entity_bias + process_bias
        if total_bias >= 0:
            continue  # no negative weight on either axis, so no conflict

        # Re-run the decision ignoring atomicity, with the combined weight added to the similarity
        probe_ctx = DecisionContext(
            threshold=threshold, process_types={process: "atomic"},
            confirmed_overrides={},
            similarity_fn=_entity_similarity_with_bias(df, {entity_pair: total_bias}),
            anchor=anchor, ignore_atomicity=True,
        )
        counterfactual = decide(process, entity, probe_ctx)

        if counterfactual.action == "ordinary_split":
            reasoning = (
                entity_reasoning.get(entity_pair, "")
                or process_reasoning.get(process_pair, "")
                or counterfactual.reasoning
            )
            conflicts.append(PendingConflict(
                # Empty when the process owns the entity itself, instead of conflicting with itself
                process=process, conflicting_process=(owner if owner != process else ""),
                entity=entity, adjusted_score=counterfactual.score, threshold=threshold,
                reasoning=reasoning,
            ))

    return conflicts


# Turns each entity group into a cluster; a process joins every cluster where it writes (C/U/D)
def _assign_process_membership(df: pd.DataFrame, ent_groups: List[List[str]]) -> List[Cluster]:
    # An atomic process without overrides writes into one group only (atomicity ran before), so it gets one cluster.
    # End-to-end processes, or atomic ones with a confirmed override, can belong to several.
    clusters = [
        Cluster(id=i, name=_cluster_name(eg), entities=list(eg))
        for i, eg in enumerate(ent_groups, start=1)
    ]
    for proc in df.index:
        for c in clusters:
            writes_here = any(_op_weight(df.at[proc, e]) > _OP_PRIORITY["R"] for e in c.entities)
            if writes_here:
                c.processes.append(proc)
    return clusters


# Builds the clusters: group entities -> enforce atomicity -> assign process membership
def _build_clusters(
    df: pd.DataFrame,
    density_threshold: float = 0.5,
    process_types: Dict[str, str] = None,
    confirmed_overrides: Dict[str, Set[str]] = None,
    entity_weights: Dict[tuple, float] = None,
) -> tuple[List[Cluster], List[Decision]]:
    # process_weights are intentionally not passed here: they do not affect cluster shape
    entity_weights = entity_weights or {}
    process_types = process_types or {}
    entities = list(df.columns)

    ent_groups = _group_entities(df, density_threshold, entity_weights)

    decisions: List[Decision] = []
    if process_types:
        ent_groups, decisions = _enforce_atomicity(
            df, entities, ent_groups, process_types, confirmed_overrides, density_threshold,
        )

    clusters = _assign_process_membership(df, ent_groups)

    return clusters, decisions


# Clusters each end-to-end (or overridden atomic) process writes to (report only)
def _compute_process_span(
    df: pd.DataFrame,
    clusters: List[Cluster],
    process_types: Dict[str, str],
    confirmed_overrides: Dict[str, Set[str]] = None,
) -> Dict[str, List[int]]:
    # Only writes count: reading data from other clusters is normal and not a cross-system responsibility
    confirmed_overrides = confirmed_overrides or {}
    span: Dict[str, List[int]] = {}
    for proc in df.index:
        is_e2e = process_types.get(proc) == "end_to_end"
        has_confirmed_split = bool(confirmed_overrides.get(proc))
        if not is_e2e and not has_confirmed_split:
            continue
        touched_ids = [
            c.id for c in clusters
            if any(_op_weight(df.at[proc, e]) > _OP_PRIORITY["R"] for e in c.entities if e in df.columns)
        ]
        span[proc] = touched_ids
    return span


# For each pure reader (no cluster), the clusters whose entities it reads (report only)
def _compute_unallocated_reads(df: pd.DataFrame, clusters: List[Cluster]) -> Dict[str, List[int]]:
    placed = {p for c in clusters for p in c.processes}
    entity_cluster: Dict[str, int] = {e: c.id for c in clusters for e in c.entities}

    reads: Dict[str, List[int]] = {}
    for proc in df.index:
        if proc in placed:
            continue
        touched_ids = sorted({
            entity_cluster[e] for e in df.columns
            if e in entity_cluster and _op_weight(df.at[proc, e]) > 0
        })
        if touched_ids:
            reads[proc] = touched_ids
    return reads


# ISA quality metrics from Vasconcelos, Sousa & Tribolet (2008); all in [0, 1], higher is better
def compute_isa_metrics(
    clusters: List[Cluster],
    crud_matrix: Dict[str, Dict[str, str]],
    process_types: Dict[str, str],
) -> Dict[str, float]:
    # RSF    : 1 = every process lives in exactly one system
    # NAIEF  : 1 = every entity is written (C/U/D) from a single system
    # LCOISF : cohesion; 1 = processes in a cluster touch the same entities (saturates near 1, compare 3rd decimal)
    # CPSMF  : 1 = atomic (critical) and end-to-end processes never share a cluster
    if not crud_matrix or not clusters:
        return {"RSF": 0.0, "NAIEF": 0.0, "LCOISF": 0.0, "CPSMF": 0.0}

    df = pd.DataFrame(crud_matrix).T
    all_procs = list(df.index)
    all_ents  = list(df.columns)

    proc_cluster_ids: Dict[str, Set[int]] = {p: set() for p in all_procs}
    for c in clusters:
        for p in c.processes:
            if p in proc_cluster_ids:
                proc_cluster_ids[p].add(c.id)

    # Where each entity is written from; a process with no cluster counts as its own separate source
    ent_cud_clusters: Dict[str, Set] = {e: set() for e in all_ents}
    for p in all_procs:
        sources = proc_cluster_ids[p] or {f"unaffiliated:{p}"}
        for e in all_ents:
            if _op_weight(df.at[p, e]) > _OP_PRIORITY["R"]:
                ent_cud_clusters[e].update(sources)

    # RSF: total processes / sum of cluster-spans per process (ideal = 1)
    total_spans = sum(max(1, len(v)) for v in proc_cluster_ids.values())
    rsf = len(all_procs) / total_spans if total_spans else 1.0

    # NAIEF: total entities / sum of CUD-cluster-spans per entity (ideal = 1)
    total_cud = sum(max(1, len(v)) for v in ent_cud_clusters.values())
    naief = len(all_ents) / total_cud if total_cud else 1.0

    # LCOISF = 1 - Σ(distinct entity-sets per cluster) / (#clusters × #processes × #entities)
    lcois_per_cluster = []
    for c in clusters:
        entity_sets_seen = set()
        for p in c.processes:
            touched = frozenset(e for e in all_ents if _op_weight(df.at[p, e]) > 0)
            entity_sets_seen.add(touched)
        lcois_per_cluster.append(len(entity_sets_seen))
    total_lcois = sum(lcois_per_cluster)
    lcoisf_denom = len(clusters) * len(all_procs) * len(all_ents)
    lcoisf = 1.0 - (total_lcois / lcoisf_denom) if lcoisf_denom else 1.0

    # CPSMF = 1 - (processes in clusters that mix critical and non-critical) / #processes
    def _is_critical(p: str) -> bool:
        return process_types.get(p, "atomic") in ("atomic", "ambiguous")

    mismatch = 0
    for c in clusters:
        has_critical = any(_is_critical(p) for p in c.processes)
        has_noncritical = any(not _is_critical(p) for p in c.processes)
        if has_critical and has_noncritical:
            mismatch += len(c.processes)
    cpsmf = 1.0 - (mismatch / len(all_procs)) if all_procs else 1.0

    return {
        "RSF":    round(rsf,    3),
        "NAIEF":  round(naief,  3),
        "LCOISF": round(lcoisf, 3),
        "CPSMF":  round(cpsmf,  3),
    }


# Converts an As-Is mapping { app: [processes] } into clusters, with entities taken from the CRUD matrix
def as_is_to_clusters(
    as_is_mapping: Dict[str, List[str]],
    crud_matrix: Dict[str, Dict[str, str]],
) -> List[Cluster]:
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


# Builds clusters from an explicit process -> system mapping (the As-Is landscape) instead of running BSP
def matrix_to_clusters(
    matrix_data: Dict[str, Dict[str, str]],
    process_system: Dict[str, str],
) -> List[Cluster]:
    df = pd.DataFrame(matrix_data).T
    systems: Dict[str, List[str]] = defaultdict(list)
    for proc, system in process_system.items():
        systems[system].append(proc)

    clusters = []
    for cid, (system, procs) in enumerate(systems.items(), start=1):
        ents = {e for p in procs for e in df.columns if _op_weight(df.at[p, e]) > 0}
        clusters.append(Cluster(id=cid, name=system, processes=list(procs), entities=list(ents)))
    return clusters


# Lists which processes/entities changed cluster between two iterations, and why
def classify_changes(
    prev_clusters: List[Cluster],
    new_clusters: List[Cluster],
    decisions: List[Decision] = None,
) -> List[ClusterChange]:
    decisions = decisions or []
    decision_by_entity = {d.entity: d for d in decisions}

    def _membership(clusters: List[Cluster]) -> Dict[str, int]:
        membership: Dict[str, int] = {}
        for c in clusters:
            for subject in c.processes + c.entities:
                membership[subject] = c.id
        return membership

    prev_membership = _membership(prev_clusters)
    new_membership = _membership(new_clusters)

    changes: List[ClusterChange] = []
    all_subjects = set(prev_membership) | set(new_membership)
    for subject in all_subjects:
        prev_id = prev_membership.get(subject)
        new_id = new_membership.get(subject)
        if prev_id == new_id:
            continue

        decision = decision_by_entity.get(subject)
        if decision and decision.action in ("force_merge", "exempt_confirmed"):
            change_type = "atomicity_override"
            reasoning = decision.reasoning
        else:
            change_type = "ordinary"
            reasoning = decision.reasoning if decision else ""

        changes.append(ClusterChange(
            subject=subject, from_cluster=prev_id, to_cluster=new_id,
            change_type=change_type, reasoning=reasoning,
        ))

    return changes


# Runs the full BSP pipeline on a CRUD matrix and returns clusters plus reporting data
def run_bsp(
    matrix_data: Dict[str, Dict[str, str]],
    density_threshold: float = 0.5,
    process_weights: Dict[tuple, float] = None,
    process_types: Dict[str, str] = None,
    confirmed_overrides: Dict[str, Set[str]] = None,
    process_reasoning: Dict[tuple, str] = None,
    adaptive_threshold: bool = False,
    adaptive_k: float = 1.0,
    adaptive_min_pairs: int = 5,
    entity_weights: Dict[tuple, float] = None,
    entity_reasoning: Dict[tuple, str] = None,
) -> BSPResult:
    # density_threshold   : minimum entity similarity to group; 0.5 follows Lee (1999)
    # process_weights     : {(proc_a, proc_b): bias} -> matrix row order + conflict detection only
    # entity_weights      : {(ent_a, ent_b): bias}   -> added to entity similarity, changes clusters
    # confirmed_overrides : {process: {entity}} the architect allowed to leave an atomic process's cluster
    # *_reasoning         : LLM justification per weight, only used to explain conflicts
    # adaptive_threshold  : replace density_threshold with derive_threshold() on this matrix
    if not matrix_data:
        raise ValueError("CRUD matrix is empty — run the extraction step first.")

    process_types = process_types or {}

    df = pd.DataFrame(matrix_data).T   # rows=processes, cols=entities

    hub_flags = _detect_hubs(df)                                                     # 1. hub detection (reporting only)
    owners    = _compute_entity_owners(df)                                           # 2. entity ownership (reporting only)
    reordered = _reorder_matrix(df, process_weights=process_weights)                 # 3. affinity reordering

    if adaptive_threshold:
        proc_scores = _pairwise_scores(
            list(reordered.index),
            lambda a, b: _weighted_jaccard(reordered, a, b, "row"),
        )
        threshold = derive_threshold(proc_scores, k=adaptive_k,
                                      min_pairs=adaptive_min_pairs, fallback=density_threshold)
    else:
        threshold = density_threshold

    clusters, decisions = _build_clusters(
        reordered, density_threshold=threshold,
        process_types=process_types, confirmed_overrides=confirmed_overrides,
        entity_weights=entity_weights,
    )                                                                                  # 4. entity groups + atomicity + membership
    pending_conflicts = _detect_pending_conflicts(
        reordered, decisions, owners, process_weights, process_reasoning, threshold,
        entity_weights=entity_weights, entity_reasoning=entity_reasoning,
    )                                                                                  # 5. atomicity-vs-constraint conflicts

    # Report only: which clusters end-to-end / overridden processes write to
    process_span = _compute_process_span(df, clusters, process_types, confirmed_overrides)

    # Report only: what pure readers (no cluster) still read
    unallocated_reads = _compute_unallocated_reads(df, clusters)

    return BSPResult(clusters=clusters, reordered_matrix=reordered,
                     entity_owners=owners, process_span=process_span,
                     hub_flags=hub_flags, pending_conflicts=pending_conflicts,
                     unallocated_reads=unallocated_reads)

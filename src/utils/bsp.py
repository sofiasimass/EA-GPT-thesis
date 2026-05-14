"""
bsp.py — IBM Business System Planning (BSP) clustering
-------------------------------------------------------
Pure algorithmic implementation of the 4-step BSP method.
No LLM involved. Takes a CRUD matrix and returns clusters.

Jaccard Similarity Coefficient (used in your _extract_blocks function)
Agglomerative Hierarchical Clustering

BSP Logic:
  Step 1 — Entity ownership: the process that CREATES an entity "owns" it.
            Ties are broken by U > R > D priority.
  Step 2 — Initial clusters: one cluster per owned-entity group.
  Step 3 — Affinity reordering: iteratively reorder rows/cols so that
            processes and entities sharing the most operations are adjacent
            (maximising block-diagonal density).
  Step 4 — Block extraction: scan the reordered matrix for contiguous dense
            blocks and name each cluster after its dominant entities.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Set
import pandas as pd
from pydantic import BaseModel, Field

#schemas

class EAWeight(BaseModel):
    item_a: str = Field(description="The name of the first process in the pair")
    item_b: str = Field(description="The name of the second process in the pair")
    weight: float = Field(description="The bias value between -0.5 and 0.5")
    reasoning: str = Field(description="Architectural justification for this weight")

class EAWeightResult(BaseModel):
    biases: List[EAWeight]


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


_OP_PRIORITY = {"C": 4, "U": 3, "R": 1, "D": 1}
_EA_WEIGHT_SCALE = 10  # Applied consistently everywhere weights are used


def _op_weight(op) -> int:
    """Numeric weight for a CRUD op cell (handles NaN and combined ops like 'CRU')."""
    if op is None or (isinstance(op, float) and pd.isna(op)):
        return 0
    return max((_OP_PRIORITY.get(ch, 0) for ch in str(op).upper()), default=0)

"""
def _affinity(df: pd.DataFrame, a: str, b: str, axis: str) -> int:
    #Count shared non-null cells between two rows (axis='row') or columns (axis='col').
    if axis == "row":
        shared = sum(
            1 for col in df.columns
            if _op_weight(df.at[a, col]) > 0 and _op_weight(df.at[b, col]) > 0
        )
    else:
        shared = sum(
            1 for row in df.index
            if _op_weight(df.at[row, a]) > 0 and _op_weight(df.at[row, b]) > 0
        )
    return shared
"""

"""
Agora considera logo os pesos e a matrix é completamente reordenada conforme os novos pesos
"""
def _affinity(df: pd.DataFrame, a: str, b: str, axis: str, ea_weights: Dict[tuple, float] = None) -> float:
    ea_weights = ea_weights or {}
    
    # Base CRUD similarity
    if axis == "row":
        shared = sum(1 for col in df.columns if _op_weight(df.at[a, col]) > 0 and _op_weight(df.at[b, col]) > 0)
    else:
        shared = sum(1 for row in df.index if _op_weight(df.at[row, a]) > 0 and _op_weight(df.at[row, b]) > 0)
    
    # Add architectural bias to the affinity score
    pair = tuple(sorted([a, b]))
    weight = ea_weights.get(pair, 0)
    
    # Scale weight so it has enough gravity to shift greedy ordering
    return shared + (weight * _EA_WEIGHT_SCALE)

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


def _initial_clusters(df: pd.DataFrame, owners: Dict[str, str]) -> List[Cluster]:
    """
    Group entities by owning process → one cluster per owning process.
    Non-owning processes are assigned to the cluster they interact with most.
    """
    owner_to_ents: Dict[str, List[str]] = {}
    for entity, owner in owners.items():
        owner_to_ents.setdefault(owner, []).append(entity)

    clusters: List[Cluster] = []
    for cid, (owner, ents) in enumerate(owner_to_ents.items(), start=1):
        clusters.append(Cluster(id=cid, name=_cluster_name(ents),
                                processes=[owner], entities=list(ents)))

    owning = {c.processes[0] for c in clusters}
    for proc in df.index:
        if proc in owning:
            continue
        best_c, best_n = clusters[0], -1
        for c in clusters:
            n = sum(1 for e in c.entities if _op_weight(df.at[proc, e]) > 0)
            if n > best_n:
                best_n, best_c = n, c
        best_c.processes.append(proc)

    return clusters


def _reorder_matrix(df: pd.DataFrame, ea_weights: Dict[tuple, float] = None) -> pd.DataFrame:
    # Pass weight-aware affinity functions to the greedy reorderer
    new_procs = _greedy_reorder(
        list(df.index), 
        lambda a, b: _affinity(df, a, b, "row", ea_weights)
    )
    new_ents  = _greedy_reorder(
        list(df.columns), 
        lambda a, b: _affinity(df, a, b, "col", ea_weights)
    )
    return df.loc[new_procs, new_ents]

def _extract_blocks(df: pd.DataFrame, density_threshold: float = 0.5, ea_weights: Dict[tuple, float] = None) -> List[Cluster]:
    """
    Scan the reordered matrix for contiguous dense rectangular blocks.
    Adjacent processes are merged into the same cluster when their entity
    overlap exceeds `density_threshold` (Jaccard similarity).
    """
    ea_weights = ea_weights or {}
    presence = df.map(lambda v: 0 if _op_weight(v) == 0 else 1)
    procs    = list(df.index)
    entities = list(df.columns)
    n_e      = len(entities)

    """
    # --- Group processes into bands ---
    proc_groups: List[List[str]] = [[procs[0]]]
    for i in range(1, len(procs)):
        prev = proc_groups[-1][-1]
        curr = procs[i]
        prev_set = {entities[j] for j in range(n_e) if presence.at[prev, entities[j]]}
        curr_set = {entities[j] for j in range(n_e) if presence.at[curr, entities[j]]}
        union = len(prev_set | curr_set)
        jaccard = len(prev_set & curr_set) / union if union else 0
        pair = tuple(sorted([prev, curr]))
        weight = ea_weights.get(pair, 0)
        similarity = jaccard + weight
        print(f"similarity of {pair} is {similarity}")
        if similarity >= density_threshold:
            print("JOINED TO CURRENT GROUP")
            proc_groups[-1].append(curr)
        else:
            print("STARTED NEW GROUP")
            proc_groups.append([curr])
    """

    # --- Group processes (considera a similarity do grupo inteiro em vez de apenas do ultimo elemento) ---
    proc_groups: List[List[str]] = [[procs[0]]]
    for i in range(1, len(procs)):
        curr = procs[i]
        current_group = proc_groups[-1]
        
        # Calculate the average similarity of 'curr' against EVERY member of the current group
        group_similarities = []
        curr_set = {entities[j] for j in range(n_e) if presence.at[curr, entities[j]]}
        
        for member in current_group:
            member_set = {entities[j] for j in range(n_e) if presence.at[member, entities[j]]}
            union = len(member_set | curr_set)
            jaccard = len(member_set & curr_set) / union if union else 0
            
            # Factor in EA weights with the same scale used in _affinity
            pair = tuple(sorted([member, curr]))
            weight = ea_weights.get(pair, 0)
            group_similarities.append(jaccard + (weight * _EA_WEIGHT_SCALE))
        
        # Use the mean similarity for the threshold check
        avg_similarity = sum(group_similarities) / len(group_similarities)
        
        print(f"Average similarity of {curr} to group {current_group} is {avg_similarity:.2f}")
        
        if avg_similarity >= density_threshold:
            print("JOINED TO CURRENT GROUP")
            proc_groups[-1].append(curr)
        else:
            print("STARTED NEW GROUP")
            proc_groups.append([curr])

    # --- Group entities into bands ---
    # Note: ea_weights keys are (process, process) pairs — they do not apply here.
    # Entity adjacency is determined purely by Jaccard similarity over shared processes.
    ent_groups: List[List[str]] = [[entities[0]]]
    for j in range(1, len(entities)):
        prev = ent_groups[-1][-1]
        curr = entities[j]
        prev_set = {p for p in procs if presence.at[p, prev]}
        curr_set = {p for p in procs if presence.at[p, curr]}
        union = len(prev_set | curr_set)
        jaccard = len(prev_set & curr_set) / union if union else 0
        if jaccard >= density_threshold:
            ent_groups[-1].append(curr)
        else:
            ent_groups.append([curr])

    # --- Pair process groups ↔ entity groups by maximum interaction count ---
    # If two process groups map to the same entity group, merge them into one cluster.
    clusters: List[Cluster] = []
    for cid, pg in enumerate(proc_groups, start=1):
        best_eg, best_score = ent_groups[0], -1
        for eg in ent_groups:
            score = int(presence.loc[pg, eg].values.sum())
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

    # --- Assign any unclaimed entity groups to the cluster that interacts with them most ---
    claimed = {e for c in clusters for e in c.entities}
    for eg in ent_groups:
        if any(e not in claimed for e in eg):
            best_c, best_score = clusters[0], -1
            for c in clusters:
                score = int(presence.loc[c.processes, eg].values.sum())
                if score > best_score:
                    best_score, best_c = score, c
            if best_score > 0:
                new_entities = [e for e in eg if e not in claimed]
                best_c.entities.extend(new_entities)
                best_c.name = _cluster_name(best_c.entities)
                claimed.update(new_entities)

    return clusters

def _absorb_pure_readers(clusters: List[Cluster], df: pd.DataFrame) -> List[Cluster]:
    """
    Any process with zero C/U operations is a pure consumer.
    Move it into whichever cluster owns (C) the most entities it reads.
    """
    owner_cluster = {}  # entity → cluster index
    for i, c in enumerate(clusters):
        for proc in c.processes:
            for ent in df.columns:
                if _op_weight(df.at[proc, ent]) == _OP_PRIORITY["C"]:
                    owner_cluster[ent] = i

    for c in clusters:
        readers_to_move = []
        for proc in c.processes:
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

    # Remove any now-empty clusters
    return [c for c in clusters if c.processes]

def run_bsp(matrix_data: Dict[str, Dict[str, str]], density_threshold: float = 0.5, ea_weights: Dict[tuple, float] = None) -> BSPResult:
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

    df = pd.DataFrame(matrix_data).T   # rows=processes, cols=entities

    owners    = _compute_entity_owners(df)           # Step 1
    _          = _initial_clusters(df, owners)        # Step 2 (scaffold only)
    reordered = _reorder_matrix(df, ea_weights=ea_weights)                   # Step 3
    clusters  = _extract_blocks(reordered, density_threshold = density_threshold, ea_weights = ea_weights)            # Step 4
    final = _absorb_pure_readers(clusters=clusters, df=df)

    return BSPResult(clusters=final, reordered_matrix=reordered, entity_owners=owners)


def print_bsp_result(result: BSPResult) -> None:
    """Pretty-print BSP results to stdout."""
    sep = "═" * 62
    print(f"\n{sep}")
    print("  BSP CLUSTERING RESULT")
    print(sep)

    print("\nEntity Ownership  (C > U > R > D priority):")
    for entity, owner in result.entity_owners.items():
        print(f"   {entity:35s} ← {owner}")

    print(f"\n {len(result.clusters)} Application Cluster(s) Identified:\n")
    for c in result.clusters:
        print(c.summary())

    print("\n Reordered CRUD Matrix:")
    print(result.reordered_matrix.fillna("·").to_string())
    print(f"{sep}\n")
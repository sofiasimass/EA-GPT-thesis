"""
bsp.py — IBM Business System Planning (BSP)-inspired clustering
-------------------------------------------------------
Pure algorithmic implementation. No LLM involved. Takes a CRUD matrix and
returns clusters.

Techniques used:
  - Weighted Jaccard similarity (process and entity grouping)
  - Greedy nearest-neighbour reordering (affinity-based)
  - Union-Find for atomic co-location constraint

Correspondence to the classical 4-step BSP method (delete irrelevant
rows/cols; group processes that create the same entities; merge in
processes that update those entities; identify clusters as simple as
possible) — this is an adaptation, not a literal step-for-step port:
  - Classical step 1 (delete irrelevant/similar rows or columns) has no
    preprocessing equivalent here — the full matrix is always reordered
    and clustered. The nearest relative is _absorb_pure_readers, which
    runs *after* clustering and relocates stray processes into an
    existing cluster rather than deleting anything up front.
  - Classical steps 2 and 3 (group processes that create the same
    entities, then fold in processes that update them) are not two
    sequential passes here. They're collapsed into one weighted-affinity
    calculation (_affinity / _weighted_jaccard), where C/U/D/R
    operations are compared via min(op_weight) — a create/update pair on
    the same entity scores nearly as high as a create/create pair, so
    creators and updaters gravitate together in a single pass.
  - Classical step 4 (identify application clusters, kept as simple as
    possible) maps to _extract_blocks (grouping adjacent processes/
    entities above density_threshold) followed by _absorb_pure_readers
    (folding stray read-only processes into an existing cluster instead
    of leaving the result fragmented).

Pipeline (see run_bsp):
  1. _compute_entity_owners — resolve which process "owns" each entity
     (the creator; ties broken by C > U > D > R priority). Used for
     reporting (BSPResult.entity_owners); does not drive the clustering.
  2. _reorder_matrix — greedy nearest-neighbour reordering of rows/columns
     by affinity, maximising block-diagonal density.
  3. _extract_blocks — group adjacent processes by weighted Jaccard
     similarity, group entities likewise, then enforce constraints:
       • Atomic processes: all entities they WRITE (C/U/D) must land in
         the same cluster (ACID — writes cannot span two systems).
         Entities they only READ may live in other clusters.
       • E2E processes: their entities may be split across clusters; the
         process itself is recorded as spanning those clusters,
         representing cross-system integration dependencies. A confirmed
         override (see below) creates the same kind of span for an
         otherwise-atomic process: it keeps ownership of the cluster
         holding what it CREATES, and the other cluster it still writes to
         is recorded as an access, not a second ownership.
  4. _absorb_pure_readers — relocate any non-E2E process with zero C/U/D
     operations into whichever cluster owns most of what it reads.

Since the restructure below, the pipeline also has optional, opt-in
extras — none change default behaviour: hub detection (BSPResult.hub_flags,
reporting only), an adaptive similarity threshold (run_bsp(adaptive_threshold=True),
replacing the fixed density_threshold with one derived from this matrix's
own similarity distribution), and an explicit decision table (decide())
that lets an architect's confirmed constraint exempt a specific entity from
an atomic process's forced union — surfaced as BSPResult.pending_conflicts
before anything is applied. classify_changes() is exported for comparing
two iterations' clusters, but isn't called from run_bsp itself.

Two independent axes of architect bias exist: process_weights (a pair of
PROCESSES, feeds _group_processes and reordering) and entity_weights (a
pair of ENTITIES, feeds _group_entities directly). Neither can silently
break an atomic process's forced write co-location — a conflict between
either axis and atomicity only ever surfaces as a PendingConflict, never
applied without confirmation.

process_type values (set by the LLM during extraction):
  "atomic"     — single indivisible transaction; cluster boundaries drawn tightly.
  "end_to_end" — spans multiple departments; cluster boundaries may split it.
  "ambiguous"  — treated as atomic (safe default).
"""

from __future__ import annotations
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Literal, Set
import statistics
import pandas as pd
from pydantic import BaseModel, Field

#schemas

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


# ---- estruturas de dados do clustering ----

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
    process_span: Dict[str, List[int]] = field(default_factory=dict)  # process → cluster ids it spans (e2e or confirmed override)
    hub_flags: List[HubFlag] = field(default_factory=list)                 # processos/entidades que dominam a matriz (só alerta)
    pending_conflicts: List[PendingConflict] = field(default_factory=list)  # conflitos atomicidade vs. pesos, por confirmar


@dataclass
class HubFlag:
    """
    Um processo ou entidade que toca em mais de metade do outro eixo da
    matriz é um "hub" — pode dominar as comparações de similaridade e
    distorcer o agrupamento à volta dele. Por agora isto só serve para
    registo/alerta, não muda o resultado do clustering.

    Bureš, Cerny, Frajtak & Ahmed, "Testing the Consistency of Business
    Data Objects Using Extended Static Testing of CRUD Matrices," Cluster
    Computing 22(S4), S963-S976, 2019.
    """
    subject: str        # nome do processo ou da entidade
    axis: str            # "process" ou "entity"
    coverage: float       # fração do outro eixo que este item toca, entre 0 e 1
    threshold: float


@dataclass
class Decision:
    """
    Uma decisão de juntar ou separar uma entidade, com a razão. Cada vez
    que o pipeline decide onde uma entidade fica, isto fica guardado aqui
    — serve para explicar ao arquiteto porque é que algo aconteceu, não só
    mostrar o resultado final.
    """
    subject: str        # processo
    entity: str          # entidade envolvida nesta decisão
    action: str           # "exempt_confirmed" | "force_merge" | "ordinary_merge" | "ordinary_split"
    reasoning: str
    score: float
    threshold: float


@dataclass
class DecisionContext:
    """Tudo o que a função decide() precisa para decidir uma entidade."""
    threshold: float
    process_types: Dict[str, str]
    confirmed_overrides: Dict[str, Set[str]]
    similarity_fn: Callable[[str, str], float]
    anchor: str                          # entidade de referência do processo (a primeira que ele escreve)
    ignore_atomicity: bool = False       # True só quando estamos a testar "e se este processo não fosse atómico?"


@dataclass
class PendingConflict:
    """
    Quando um processo atómico é obrigado a juntar-se a uma entidade, mas o
    peso dado pelo arquiteto diz que deviam estar separados — isto fica
    registado aqui. Nunca é aplicado sozinho, só é mostrado para
    confirmação (ver confirmed_overrides em run_bsp).
    """
    process: str
    conflicting_process: str
    entity: str
    adjusted_score: float
    threshold: float
    reasoning: str


@dataclass
class ClusterChange:
    """Diferença entre o clustering de uma iteração e o da iteração anterior."""
    subject: str
    from_cluster: int | None
    to_cluster: int | None
    change_type: str     # "atomicity_override" ou "ordinary"
    reasoning: str


# Numeric weights for CRUD operations.
# C=4 (highest) because creation implies full ownership of the entity.
# Used in affinity calculations and entity ownership resolution.
_OP_PRIORITY = {"C": 4, "U": 3, "D": 2, "R": 1}


def _op_weight(op) -> int:
    """Numeric weight for a CRUD op cell (handles NaN and combined ops like 'CRU')."""
    if op is None or (isinstance(op, float) and pd.isna(op)):
        return 0
    return max((_OP_PRIORITY.get(ch, 0) for ch in str(op).upper()), default=0)


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


# ---- deteção de hubs ----

def _detect_hubs(df: pd.DataFrame, threshold: float = 0.5) -> List[HubFlag]:
    """
    Um processo ou entidade que toca em mais de `threshold` do outro eixo é
    um "hub" — pode puxar coisas que não têm nada a ver umas com as outras
    para o mesmo cluster, só por estar em todo o lado. Por agora só regista
    o alerta em BSPResult.hub_flags, não muda o agrupamento (isso fica
    para depois de testar, como o adaptive_threshold).

    Corre antes do _reorder_matrix, sobre a matriz original — reordenar
    não muda quantas colunas/linhas um item toca, mas a citação que
    justifica isto fala em fazer a deteção antes de reordenar.

    Bureš, Cerny, Frajtak & Ahmed, "Testing the Consistency of Business
    Data Objects Using Extended Static Testing of CRUD Matrices," Cluster
    Computing 22(S4), S963-S976, 2019.
    """
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


def _affinity(df: pd.DataFrame, a: str, b: str, axis: str, process_weights: Dict[tuple, float] = None) -> float:
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

    process_weights (architect-supplied biases in [-0.5, 0.5]) are added directly on top,
    allowing targeted boosts or penalties without a global scaling constant.
    """
    process_weights = process_weights or {}
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

    # Bias arquitetural: somado diretamente — weights em [-0.5, 0.5] têm
    # impacto proporcional sobre uma afinidade entre [0, 1].
    pair = tuple(sorted([a, b]))
    weight = process_weights.get(pair, 0)

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


def _reorder_matrix(df: pd.DataFrame, process_weights: Dict[tuple, float] = None) -> pd.DataFrame:
    """
    Reorder rows (processes) and columns (entities) so that items with high
    mutual affinity end up adjacent, producing a block-diagonal structure.
    process_weights shift affinity scores before reordering, so architect-supplied
    biases influence the final layout directly.
    """
    new_procs = _greedy_reorder(
        list(df.index),
        lambda a, b: _affinity(df, a, b, "row", process_weights)
    )
    new_ents  = _greedy_reorder(
        list(df.columns),
        lambda a, b: _affinity(df, a, b, "col", process_weights)
    )
    return df.loc[new_procs, new_ents]


# ---- pipeline de clustering ----
# A antiga _extract_blocks fazia tudo numa função só (agrupar processos,
# agrupar entidades, aplicar a regra da atomicidade, e juntar tudo em
# clusters). Agora cada passo é uma função com um nome e uma
# responsabilidade — mais fácil de perceber, de testar, e de estender
# (foi aqui que entrou o threshold adaptativo e a tabela de decisão).

def _weighted_jaccard(df: pd.DataFrame, a: str, b: str, axis: str) -> float:
    """
    Jaccard ponderado entre dois processos (axis="row") ou duas entidades
    (axis="col"). Substitui as duas funções que existiam antes
    (_weighted_jaccard_procs e _weighted_jaccard_ents) por uma só, com o
    mesmo "axis" que a função _affinity já usa.

    Não é a mesma coisa que _affinity: aqui a normalização é pela soma real
    da interseção/união deste par (dá "quanta sobreposição têm estes dois,
    dos dois"), enquanto _affinity normaliza por uma constante fixa
    (4 × número de itens) porque serve para ordenar a matriz toda, não
    para comparar com um threshold. Por isso não foram fundidas numa só.
    """
    if axis == "row":
        others = df.columns
        intersection = sum(min(_op_weight(df.at[a, o]), _op_weight(df.at[b, o])) for o in others)
        union_sum    = sum(max(_op_weight(df.at[a, o]), _op_weight(df.at[b, o])) for o in others)
    else:
        others = df.index
        intersection = sum(min(_op_weight(df.at[o, a]), _op_weight(df.at[o, b])) for o in others)
        union_sum    = sum(max(_op_weight(df.at[o, a]), _op_weight(df.at[o, b])) for o in others)

    return intersection / union_sum if union_sum else 0


def _entity_similarity_with_bias(df: pd.DataFrame, entity_weights: Dict[tuple, float] = None) -> Callable[[str, str], float]:
    """
    Uma função de similaridade entre duas entidades que já soma o peso do
    arquiteto ao nível das entidades (entity_weights), quando existir para
    aquele par — o mesmo princípio do bias em _affinity/_group_processes,
    mas para o eixo das entidades, e já pronta a usar como
    DecisionContext.similarity_fn (que só aceita uma função, sem bias_fn
    à parte).

    Usada em _detect_pending_conflicts: antes disto existir, o único sinal
    negativo possível vinha de um peso entre dois PROCESSOS (via o dono da
    entidade) — um proxy indireto. Agora uma restrição do arquiteto pode
    dizer diretamente "estas duas entidades não podem estar juntas", sem
    precisar de passar por que processo é que cria cada uma.
    """
    entity_weights = entity_weights or {}

    def fn(a: str, b: str) -> float:
        return _weighted_jaccard(df, a, b, "col") + entity_weights.get(tuple(sorted([a, b])), 0)

    return fn


def _average_linkage_group(
    items: List[str],
    similarity_fn: Callable[[str, str], float],
    threshold: float,
    bias_fn: Callable[[str, str], float] = None,
) -> List[List[str]]:
    """
    Junta itens em grupos, sempre a comparar com a média do grupo todo até
    agora — não só com o último item que entrou. Isto evita o problema da
    "corrente": se A e B são parecidos, e B e C são parecidos, A e C podem
    não ter nada a ver um com o outro. Comparar só com o último deixava
    isso passar; comparar com a média do grupo inteiro não deixa.

    bias_fn é opcional — só é usado para agrupar processos (os pesos do
    arquiteto), nunca para entidades.
    """
    if not items:
        return []

    groups: List[List[str]] = [[items[0]]]
    for curr in items[1:]:
        current_group = groups[-1]
        similarities = []
        for member in current_group:
            score = similarity_fn(curr, member)
            if bias_fn:
                score += bias_fn(curr, member)
            similarities.append(score)

        avg_similarity = sum(similarities) / len(similarities)

        if avg_similarity >= threshold:
            groups[-1].append(curr)
        else:
            groups.append([curr])

    return groups


def _group_processes(df: pd.DataFrame, threshold: float, process_weights: Dict[tuple, float] = None) -> List[List[str]]:
    """Agrupa processos, com os pesos do arquiteto a poder influenciar o resultado."""
    process_weights = process_weights or {}

    def bias(a: str, b: str) -> float:
        return process_weights.get(tuple(sorted([a, b])), 0)

    return _average_linkage_group(
        list(df.index),
        lambda a, b: _weighted_jaccard(df, a, b, "row"),
        threshold,
        bias_fn=bias,
    )


def _group_entities(df: pd.DataFrame, threshold: float, entity_weights: Dict[tuple, float] = None) -> List[List[str]]:
    """
    Agrupa entidades, com pesos do arquiteto ao nível das entidades
    (entity_weights) a poder influenciar o resultado — o mesmo princípio
    do _group_processes, mas no eixo das entidades. Antes disto existir,
    o agrupamento de entidades usava só o sinal estrutural do Jaccard,
    porque o único peso que existia (process_weights) é sobre pares de
    processos e não faz sentido aqui. Agora uma restrição do tipo "estas
    duas entidades têm de ficar em sistemas separados" pode agir
    diretamente neste passo, em vez de só ao nível dos processos.
    """
    entity_weights = entity_weights or {}

    def bias(a: str, b: str) -> float:
        return entity_weights.get(tuple(sorted([a, b])), 0)

    return _average_linkage_group(
        list(df.columns),
        lambda a, b: _weighted_jaccard(df, a, b, "col"),
        threshold,
        bias_fn=bias,
    )


def _pairwise_scores(items: List[str], similarity_fn: Callable[[str, str], float]) -> List[float]:
    """Todos os valores de similaridade par a par desta lista, para calcular estatísticas em cima deles."""
    scores = []
    for i, a in enumerate(items):
        for b in items[i + 1:]:
            scores.append(similarity_fn(a, b))
    return scores


def derive_threshold(
    scores: List[float],
    k: float = 1.0,
    min_pairs: int = 5,
    fallback: float = 0.5,
) -> float:
    """
    Threshold adaptativo: média + k × desvio-padrão da distribuição de
    similaridades desta matriz, em vez de um número fixo escolhido à
    partida sem saber como os dados iam ser. Um threshold fixo (como 0.5)
    pode não fazer sentido para uma matriz muito "solta" (quase tudo fica
    sozinho, ninguém chega a 0.5) ou muito densa (quase tudo se junta) —
    este adapta-se aos dados reais de cada matriz.

    Se não houver pares suficientes (`min_pairs`) para uma estatística com
    significado, usa `fallback` (o antigo valor fixo) em vez de arriscar.

    Akkasi, Seyyedi & Shams, "Presenting A Method for Benchmarking
    Application in the Enterprise Architecture Planning Process Based on
    Federal Enterprise Architecture Framework," IEEE Xplore.

    NOTA: o k=2 do artigo original é para outro tipo de problema (aceitar
    ou não um pequeno conjunto de candidatos face a um único parceiro
    ideal). Aqui o threshold é usado em muitas comparações seguidas
    enquanto se percorre a matriz toda, por isso k=1.0 é só um ponto de
    partida — precisa de ser testado e ajustado com matrizes reais antes
    de confiar nele.
    """
    if len(scores) < min_pairs:
        return fallback
    mean = statistics.fmean(scores)
    stdev = statistics.pstdev(scores)
    return mean + k * stdev


class _UnionFind:
    """
    Union-find simples: cada entidade começa sozinha num grupo, e vamos
    juntando (union) as que têm de ficar juntas. No fim, groups() devolve
    os grupos finais, pela ordem em que as entidades apareciam.
    """
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

    def groups(self, order: List[str]) -> List[List[str]]:
        buckets: Dict[str, List[str]] = defaultdict(list)
        for item in order:
            buckets[self.find(item)].append(item)

        order_index = {item: i for i, item in enumerate(order)}
        result = list(buckets.values())
        result.sort(key=lambda g: order_index[g[0]])
        return result


def decide(process: str, entity: str, ctx: DecisionContext) -> Decision:
    """
    A tabela de decisão: para cada par processo/entidade, decide o que
    fazer — e regista qual foi a razão. As regras têm uma ordem de
    prioridade explícita, em vez de estarem escondidas na ordem do código
    como antes:

      1. Se o arquiteto já confirmou que esta entidade pode sair do
         cluster forçado pelo processo atómico — ganha sempre, seja qual
         for o resto.
      2. Se o processo é atómico e escreve nesta entidade — junta à força
         (é a regra ACID: uma transação não pode ficar espalhada por dois
         sistemas).
      3. Senão, é só a comparação normal: a similaridade chega ao
         threshold ou não chega.

    Grad, B., "Decision Tables in Systems Design," Session 19, Digest of
    Technical Papers, 1962 ACM National Conference, pp. 76-77.

    Nota: process_types.get(process, "atomic") == "atomic" é a mesma
    condição literal que já existia antes — "ambiguous" NÃO cai aqui,
    mesmo o docstring do módulo dizendo que é tratado como atómico. É uma
    inconsistência que já existia no código original; fica por resolver
    noutra altura, não é para corrigir silenciosamente nesta reestruturação.
    """
    if entity in ctx.confirmed_overrides.get(process, set()):
        return Decision(
            subject=process, entity=entity, action="exempt_confirmed",
            reasoning=f"o arquiteto confirmou que {entity} pode sair do cluster de {process}",
            score=0.0, threshold=ctx.threshold,
        )

    if not ctx.ignore_atomicity and ctx.process_types.get(process, "atomic") == "atomic":
        return Decision(
            subject=process, entity=entity, action="force_merge",
            reasoning=f"{process} é atómico e escreve em {entity} — têm de ficar no mesmo sistema",
            score=0.0, threshold=ctx.threshold,
        )

    score = ctx.similarity_fn(ctx.anchor, entity)
    if score >= ctx.threshold:
        action = "ordinary_merge"
        reasoning = f"similaridade {score:.2f} >= threshold {ctx.threshold:.2f}"
    else:
        action = "ordinary_split"
        reasoning = f"similaridade {score:.2f} < threshold {ctx.threshold:.2f}"

    return Decision(subject=process, entity=entity, action=action, reasoning=reasoning,
                     score=score, threshold=ctx.threshold)


def _enforce_atomicity(
    df: pd.DataFrame,
    entities: List[str],
    ent_groups: List[List[str]],
    process_types: Dict[str, str],
    confirmed_overrides: Dict[str, Set[str]],
    threshold: float,
) -> tuple[List[List[str]], List[Decision]]:
    """
    Aplica a regra de co-localização: um processo atómico obriga todas as
    entidades que ele escreve (C/U/D) a ficarem no mesmo grupo. Usa
    union-find porque as junções em cascata (A+B, depois B+C) têm de ser
    tratadas corretamente.

    Quando confirmed_overrides está vazio — que é o valor por omissão, e é
    o que app.py e todos os testes atuais usam — decide() nunca devolve
    "exempt_confirmed". Ou seja, isto faz exatamente o que o código antigo
    fazia, processo a processo, entidade a entidade, sem exceção.
    """
    confirmed_overrides = confirmed_overrides or {}
    uf = _UnionFind(entities, seed_groups=ent_groups)
    decisions: List[Decision] = []

    for proc in df.index:
        if process_types.get(proc, "atomic") != "atomic":
            continue

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
            # "exempt_confirmed" não junta nada — a entidade fica livre
            # para o agrupamento normal decidir para onde vai.

    new_groups = uf.groups(entities)
    return new_groups, decisions


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
    """
    Para cada decisão "force_merge", pergunta: "e se este processo não
    fosse atómico, o que é que o agrupamento normal teria decidido?" Se a
    resposta fosse "separar", há um conflito real entre a regra da
    atomicidade e o que o arquiteto está a pedir — fica registado aqui,
    mas nunca é aplicado sozinho, só depois de confirmado (ver
    confirmed_overrides em run_bsp).

    Há dois eixos de onde pode vir o sinal negativo que desencadeia esta
    verificação, e um não substitui o outro — somam-se:
      - entity_weights: peso direto entre a âncora do processo (a primeira
        entidade que ele escreve) e a entidade em causa. É o sinal mais
        direto possível — "estas duas entidades não podem estar juntas",
        sem ter de passar por que processo é que cria cada uma.
      - process_weights: peso entre o processo atómico e o processo dono
        da entidade em causa (via entity_owners) — o sinal antigo, mantido
        porque por agora (antes da Fase 2) é o único que o prompt da LLM
        sabe produzir. Só se aplica quando a entidade pertence a OUTRO
        processo — não faz sentido "entrar em conflito" com uma entidade
        que o próprio processo criou.

    Se não há nenhum dos dois pesos, não pode haver conflito nenhum — por
    isso devolve logo lista vazia, sem correr o resto da função.

    A comparação de entidades (_weighted_jaccard) nunca usa nenhum destes
    pesos por si só — essa regra não mudou. Por isso, para esta
    verificação em concreto, soma-se o peso total (dos dois eixos) à
    similaridade, só durante este teste — sem isso, o resultado seria
    sempre o mesmo com ou sem pesos, e nunca refletiria o que o arquiteto
    pediu.
    """
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

        owner = entity_owners.get(entity, "")
        process_pair = tuple(sorted([process, owner]))
        process_bias = process_weights.get(process_pair, 0) if owner and owner != process else 0

        total_bias = entity_bias + process_bias
        if total_bias >= 0:
            continue  # sem peso negativo em nenhum dos eixos, não há sinal de conflito

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
                # Se a entidade é dona do próprio processo (o mesmo processo
                # cria as duas entidades em conflito), não há "outro processo"
                # a apontar — fica vazio, em vez de dizer que P está em
                # conflito consigo mesmo.
                process=process, conflicting_process=(owner if owner != process else ""),
                entity=entity, adjusted_score=counterfactual.score, threshold=threshold,
                reasoning=reasoning,
            ))

    return conflicts


def _effective_op_weight(df: pd.DataFrame, p: str, e: str, confirmed_overrides: Dict[str, Set[str]]) -> int:
    """
    Igual ao _op_weight, mas devolve 0 se esta entidade já foi confirmada
    como isenta deste processo (ver confirmed_overrides). Serve só para
    _assign_unclaimed_entities — sem isto, um grupo de entidades que o
    _enforce_atomicity já separou com sucesso podia voltar a ser colado ao
    cluster do processo de onde saiu, só porque este passo recalcula o
    score do zero e não sabe que essa separação já tinha sido pedida de
    propósito. (Já não é usado em _pair_groups_to_clusters — ver o
    comentário lá para o porquê.)
    """
    if e in (confirmed_overrides or {}).get(p, set()):
        return 0
    return _op_weight(df.at[p, e])


def _pair_groups_to_clusters(
    df: pd.DataFrame,
    proc_groups: List[List[str]],
    ent_groups: List[List[str]],
) -> List[Cluster]:
    """
    Junta cada grupo de processos ao grupo de entidades com que mais
    interage (score = soma dos pesos CRUD reais). Se dois grupos de
    processos calharem no mesmo grupo de entidades, ficam no mesmo cluster.

    Usa _op_weight (não _effective_op_weight) de propósito: um
    confirmed_override tira uma entidade da união forçada da atomicidade,
    mas não tira a posse — o processo continua a ser quem CRIA aquela
    entidade. Zerar essa relação aqui fazia o processo "perder a casa"
    certa (a entidade que ele cria) e ser encaixado onde só faz um UPDATE
    mais fraco, só porque o score da entidade certa tinha sido zerado.
    Encontrado ao vivo: "Purchase order creation and approvals" (cria
    Purchase Order, atualiza Budget) devia ficar no cluster do Purchase
    Order, mas com o score zerado acabava sempre no cluster do Budget. A
    ligação a Budget não desaparece — passa a ser reportada como acesso
    cross-cluster por _compute_process_span, não como posse aqui.
    """
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
    return clusters


def _assign_unclaimed_entities(
    clusters: List[Cluster],
    ent_groups: List[List[str]],
    df: pd.DataFrame,
    confirmed_overrides: Dict[str, Set[str]] = None,
) -> List[Cluster]:
    """
    Grupos de entidades que ainda não ficaram em nenhum cluster vão para o
    cluster com que mais se relacionam — outra vez ignorando pares já
    confirmados como isentos, para não desfazer uma separação que já foi
    pedida. Se nenhum cluster existente tiver ligação real a este grupo
    (porque a única ligação era precisamente a que acabou de ser isentada),
    o grupo passa a ser o seu próprio cluster novo, em vez de desaparecer.
    """
    claimed = {e for c in clusters for e in c.entities}
    next_id = max((c.id for c in clusters), default=0) + 1
    for eg in ent_groups:
        if any(e not in claimed for e in eg):
            best_c, best_score = None, 0
            for c in clusters:
                score = sum(_effective_op_weight(df, p, e, confirmed_overrides) for p in c.processes for e in eg)
                if score > best_score:
                    best_score, best_c = score, c

            new_entities = [e for e in eg if e not in claimed]
            if best_c is not None:
                best_c.entities.extend(new_entities)
                best_c.name = _cluster_name(best_c.entities)
            else:
                clusters.append(Cluster(id=next_id, name=_cluster_name(new_entities), entities=new_entities))
                next_id += 1
            claimed.update(new_entities)
    return clusters


def _extract_blocks(
    df: pd.DataFrame,
    density_threshold: float = 0.5,
    process_weights: Dict[tuple, float] = None,
    process_types: Dict[str, str] = None,
    confirmed_overrides: Dict[str, Set[str]] = None,
    entity_weights: Dict[tuple, float] = None,
) -> tuple[List[Cluster], List[Decision]]:
    """
    Orquestra o passo 4: agrupa processos, agrupa entidades, obriga os
    processos atómicos a ficarem juntos com o que escrevem, junta tudo em
    clusters, e por fim distribui os grupos de entidades que sobraram.
    """
    process_weights = process_weights or {}
    entity_weights = entity_weights or {}
    process_types = process_types or {}
    entities = list(df.columns)

    proc_groups = _group_processes(df, density_threshold, process_weights)
    ent_groups = _group_entities(df, density_threshold, entity_weights)

    decisions: List[Decision] = []
    if process_types:
        ent_groups, decisions = _enforce_atomicity(
            df, entities, ent_groups, process_types, confirmed_overrides, density_threshold,
        )

    clusters = _pair_groups_to_clusters(df, proc_groups, ent_groups)
    clusters = _assign_unclaimed_entities(clusters, ent_groups, df, confirmed_overrides)

    return clusters, decisions


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

    # Remove only clusters that are truly empty (no processes AND no
    # entities), then renumber so IDs stay contiguous 1..N — otherwise a
    # dropped cluster leaves a gap (e.g. 1,2,3,5,6…) that every downstream
    # LLM call and the UI would otherwise have to explain away.
    #
    # `c.processes` alone used to be the filter, on the assumption that a
    # process-less cluster only ever happens when every reader that used to
    # live there got moved out above. But _assign_unclaimed_entities can
    # also produce a process-less cluster on purpose — an entity that a
    # confirmed override just pulled away from its own atomic process,
    # which scores 0 against every existing cluster and gets a brand-new
    # one with entities but no owning process. Filtering on `c.processes`
    # alone silently deleted that cluster (and its entity) here — found
    # live, an entity vanished from the final output after a confirmed
    # split, instead of ending up in its own single-entity system.
    survivors = [c for c in clusters if c.processes or c.entities]
    for new_id, c in enumerate(survivors, start=1):
        c.id = new_id
    return survivors

def _compute_process_span(
    df: pd.DataFrame,
    clusters: List[Cluster],
    process_types: Dict[str, str],
    confirmed_overrides: Dict[str, Set[str]] = None,
) -> Dict[str, List[int]]:
    """
    Para cada processo que legitimamente precisa de acesso a mais do que um
    cluster, devolve a lista de IDs de cluster que ele toca — só conta
    ESCRITAS (C/U/D), não leituras. Só para relatório — nunca muda
    process_types nem entra no cálculo do CPSMF.

    Antes disto só cobria processos end_to_end (era _compute_e2e_span).
    Generalizado para também cobrir o caso encontrado ao vivo: um processo
    atómico cria a entidade A e atualiza a entidade B; o arquiteto confirma
    que A e B devem ficar em clusters diferentes (confirmed_overrides);
    _pair_groups_to_clusters dá a esse processo o cluster que ele CRIA (A),
    mas ele continua a precisar de aceder ao cluster onde só faz UPDATE
    (B) — é a mesma dependência cross-system que já existia para
    end_to_end, só que chega lá por uma decisão confirmada, não por uma
    classificação.

    Só escritas contam de propósito: quase todos os processos LEEM coisas
    de fora do seu próprio cluster (dados de referência, catálogos) — isso
    é normal e não é um "span" digno de destaque. Contar leituras tornava
    isto ruidoso (quase tudo parecia espalhado por todo o lado); só uma
    escrita fora do cluster próprio representa mesmo uma responsabilidade
    cross-system. Mesmo threshold que _enforce_atomicity já usa para
    decidir o que um processo "escreve" (_op_weight > R).
    """
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


def classify_changes(
    prev_clusters: List[Cluster],
    new_clusters: List[Cluster],
    decisions: List[Decision] = None,
) -> List[ClusterChange]:
    """
    Compara o clustering novo com o anterior e diz o que mudou — e porquê.
    Se a mudança envolveu quebrar (ou confirmar) a atomicidade, fica
    marcada como "atomicity_override"; senão é só uma mudança normal.

    Não é chamada de dentro do run_bsp — é para o app.py usar mais tarde,
    quando quiser mostrar ao arquiteto o que mudou entre duas iterações.
    """
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
    """
    Run the full 4-step BSP algorithm.

    Parameters
    ----------
    matrix_data : { process_name: { entity_name: "C"|"R"|"U"|"D" } }
        Raw CRUD matrix dict as stored in Matrix.matrix.
    confirmed_overrides : { process_name: {entity_name, ...} }
        Entities the architect has explicitly confirmed can leave an
        atomic process's forced cluster, for this run only.
    process_reasoning : { (process_a, process_b): reasoning }
        The LLM's justification text per process_weights pair (same keys). Only
        used to explain a PendingConflict — never affects clustering.
    entity_weights : { (entity_a, entity_b): bias in [-0.5, 0.5] }
        Architect bias on a pair of ENTITIES rather than processes — feeds
        _group_entities directly, and the _detect_pending_conflicts
        counterfactual. A process caught between two entities pulled apart
        this way still can't have its atomicity silently broken; it only
        ever surfaces as a PendingConflict, same as process_weights.
    entity_reasoning : { (entity_a, entity_b): reasoning }
        The LLM's justification text per entity_weights pair (same keys).
        Only used to explain a PendingConflict — never affects clustering.
    adaptive_threshold : if True, replaces density_threshold with a value
        derived from this matrix's own similarity distribution instead of
        the fixed constant. Default False keeps today's exact behaviour.

    Returns
    -------
    BSPResult with clusters, reordered_matrix, entity_owners, process_span,
    hub_flags, pending_conflicts.
    """
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

    clusters, decisions = _extract_blocks(
        reordered, density_threshold=threshold, process_weights=process_weights,
        process_types=process_types, confirmed_overrides=confirmed_overrides,
        entity_weights=entity_weights,
    )                                                                                  # 4. block extraction + constraints
    pending_conflicts = _detect_pending_conflicts(
        reordered, decisions, owners, process_weights, process_reasoning, threshold,
        entity_weights=entity_weights, entity_reasoning=entity_reasoning,
    )                                                                                  # 5. atomicity-vs-constraint conflicts
    final    = _absorb_pure_readers(clusters=clusters, df=df,
                                     process_types=process_types)                     # 6. pure-reader cleanup
    process_span = _compute_process_span(df, final, process_types, confirmed_overrides)

    return BSPResult(clusters=final, reordered_matrix=reordered,
                     entity_owners=owners, process_span=process_span,
                     hub_flags=hub_flags, pending_conflicts=pending_conflicts)

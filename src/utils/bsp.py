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
         representing cross-system integration dependencies.
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

class EAWeight(BaseModel):
    first_process: str = Field(description="The name of the first process in the pair")
    second_process: str = Field(description="The name of the second process in the pair")
    weight: float = Field(description="The bias value between -0.5 and 0.5")
    reasoning: str = Field(description="Architectural justification for this weight")

class EAWeightResult(BaseModel):
    biases: List[EAWeight]


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
    e2e_span: Dict[str, List[int]] = field(default_factory=dict)  # e2e process → cluster ids it spans
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

    # Bias arquitetural: somado diretamente — weights em [-0.5, 0.5] têm
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


def _group_processes(df: pd.DataFrame, threshold: float, ea_weights: Dict[tuple, float] = None) -> List[List[str]]:
    """Agrupa processos, com os pesos do arquiteto a poder influenciar o resultado."""
    ea_weights = ea_weights or {}

    def bias(a: str, b: str) -> float:
        return ea_weights.get(tuple(sorted([a, b])), 0)

    return _average_linkage_group(
        list(df.index),
        lambda a, b: _weighted_jaccard(df, a, b, "row"),
        threshold,
        bias_fn=bias,
    )


def _group_entities(df: pd.DataFrame, threshold: float) -> List[List[str]]:
    """
    Agrupa entidades. Os pesos do arquiteto (ea_weights) são sobre pares de
    processos, não fazem sentido aqui — o agrupamento de entidades usa só
    o sinal estrutural do Jaccard.
    """
    return _average_linkage_group(
        list(df.columns),
        lambda a, b: _weighted_jaccard(df, a, b, "col"),
        threshold,
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
    ea_weights: Dict[tuple, float],
    ea_reasoning: Dict[tuple, str],
    threshold: float,
) -> List[PendingConflict]:
    """
    Para cada decisão "force_merge", pergunta: "e se este processo não
    fosse atómico, o que é que o agrupamento normal teria decidido?" Se a
    resposta fosse "separar", há um conflito real entre a regra da
    atomicidade e o que o peso do arquiteto está a pedir — fica registado
    aqui, mas nunca é aplicado sozinho, só depois de confirmado (ver
    confirmed_overrides em run_bsp).

    Se não há ea_weights, não pode haver conflito nenhum — por isso
    devolve logo lista vazia, sem correr o resto da função. É uma garantia
    explícita, não um acaso da matemática.

    Duas coisas a que é preciso ter cuidado aqui:
      - Uma entidade que o próprio processo criou não conta — não faz
        sentido "entrar em conflito" com uma entidade que é dele mesmo.
        Só interessa quando a entidade pertence a OUTRO processo.
      - A comparação de entidades (_weighted_jaccard) nunca usa ea_weights
        — essa regra já existia antes e não foi mudada. Por isso, para
        esta verificação em concreto, soma-se o peso do par
        (processo, dono-da-entidade) diretamente à similaridade, só
        durante este teste — sem isso, o resultado seria sempre o mesmo
        com ou sem pesos, e nunca refletiria o que o arquiteto pediu.
    """
    if not ea_weights:
        return []

    ea_reasoning = ea_reasoning or {}
    conflicts: List[PendingConflict] = []

    for decision in decisions:
        if decision.action != "force_merge":
            continue

        process = decision.subject
        entity = decision.entity
        owner = entity_owners.get(entity, "")
        if not owner or owner == process:
            continue

        pair = tuple(sorted([process, owner]))
        weight = ea_weights.get(pair, 0)
        if weight >= 0:
            continue  # sem peso negativo para este par, não há sinal de conflito

        touched = [e for e in df.columns if _op_weight(df.at[process, e]) > _OP_PRIORITY["R"]]
        if not touched:
            continue
        anchor = touched[0]

        def _biased_similarity(a: str, b: str, _w=weight) -> float:
            return _weighted_jaccard(df, a, b, "col") + _w

        probe_ctx = DecisionContext(
            threshold=threshold, process_types={process: "atomic"},
            confirmed_overrides={},
            similarity_fn=_biased_similarity,
            anchor=anchor, ignore_atomicity=True,
        )
        counterfactual = decide(process, entity, probe_ctx)

        if counterfactual.action == "ordinary_split":
            owner = entity_owners.get(entity, "")
            reasoning = ea_reasoning.get(tuple(sorted([process, owner])), "") or counterfactual.reasoning
            conflicts.append(PendingConflict(
                process=process, conflicting_process=owner, entity=entity,
                adjusted_score=counterfactual.score, threshold=threshold,
                reasoning=reasoning,
            ))

    return conflicts


def _pair_groups_to_clusters(df: pd.DataFrame, proc_groups: List[List[str]], ent_groups: List[List[str]]) -> List[Cluster]:
    """
    Junta cada grupo de processos ao grupo de entidades com que mais
    interage (score = soma dos pesos CRUD). Se dois grupos de processos
    calharem no mesmo grupo de entidades, ficam no mesmo cluster.
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


def _assign_unclaimed_entities(clusters: List[Cluster], ent_groups: List[List[str]], df: pd.DataFrame) -> List[Cluster]:
    """Grupos de entidades que ainda não ficaram em nenhum cluster vão para o cluster com que mais se relacionam."""
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


def _extract_blocks(
    df: pd.DataFrame,
    density_threshold: float = 0.5,
    ea_weights: Dict[tuple, float] = None,
    process_types: Dict[str, str] = None,
    confirmed_overrides: Dict[str, Set[str]] = None,
) -> tuple[List[Cluster], List[Decision]]:
    """
    Orquestra o passo 4: agrupa processos, agrupa entidades, obriga os
    processos atómicos a ficarem juntos com o que escrevem, junta tudo em
    clusters, e por fim distribui os grupos de entidades que sobraram.
    """
    ea_weights = ea_weights or {}
    process_types = process_types or {}
    entities = list(df.columns)

    proc_groups = _group_processes(df, density_threshold, ea_weights)
    ent_groups = _group_entities(df, density_threshold)

    decisions: List[Decision] = []
    if process_types:
        ent_groups, decisions = _enforce_atomicity(
            df, entities, ent_groups, process_types, confirmed_overrides, density_threshold,
        )

    clusters = _pair_groups_to_clusters(df, proc_groups, ent_groups)
    clusters = _assign_unclaimed_entities(clusters, ent_groups, df)

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
    ea_weights: Dict[tuple, float] = None,
    process_types: Dict[str, str] = None,
    confirmed_overrides: Dict[str, Set[str]] = None,
    ea_reasoning: Dict[tuple, str] = None,
    adaptive_threshold: bool = False,
    adaptive_k: float = 1.0,
    adaptive_min_pairs: int = 5,
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
    ea_reasoning : { (process_a, process_b): reasoning }
        The LLM's justification text per ea_weights pair (same keys). Only
        used to explain a PendingConflict — never affects clustering.
    adaptive_threshold : if True, replaces density_threshold with a value
        derived from this matrix's own similarity distribution instead of
        the fixed constant. Default False keeps today's exact behaviour.

    Returns
    -------
    BSPResult with clusters, reordered_matrix, entity_owners, e2e_span,
    hub_flags, pending_conflicts.
    """
    if not matrix_data:
        raise ValueError("CRUD matrix is empty — run the extraction step first.")

    process_types = process_types or {}

    df = pd.DataFrame(matrix_data).T   # rows=processes, cols=entities

    hub_flags = _detect_hubs(df)                                                     # 1. hub detection (reporting only)
    owners    = _compute_entity_owners(df)                                           # 2. entity ownership (reporting only)
    reordered = _reorder_matrix(df, ea_weights=ea_weights)                           # 3. affinity reordering

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
        reordered, density_threshold=threshold, ea_weights=ea_weights,
        process_types=process_types, confirmed_overrides=confirmed_overrides,
    )                                                                                  # 4. block extraction + constraints
    pending_conflicts = _detect_pending_conflicts(
        reordered, decisions, owners, ea_weights, ea_reasoning, threshold,
    )                                                                                  # 5. atomicity-vs-constraint conflicts
    final    = _absorb_pure_readers(clusters=clusters, df=df,
                                     process_types=process_types)                     # 6. pure-reader cleanup
    e2e_span  = _compute_e2e_span(df, final, process_types)

    return BSPResult(clusters=final, reordered_matrix=reordered,
                     entity_owners=owners, e2e_span=e2e_span,
                     hub_flags=hub_flags, pending_conflicts=pending_conflicts)

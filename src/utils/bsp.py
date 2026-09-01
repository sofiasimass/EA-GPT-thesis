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
    equivalent here — the full matrix is always reordered and clustered.
    A process with zero writes simply ends up a member of no cluster at
    all (see below) instead of being deleted from the matrix up front.
  - Classical steps 2 and 3 (group processes that create the same
    entities, then fold in processes that update them) collapse into a
    single rule: a cluster IS an entity group (_group_entities), and a
    process is a member of every cluster it genuinely WRITES to
    (_assign_process_membership) — no separate "who owns this cluster"
    contest, a process can legitimately belong to more than one.
  - Classical step 4 (identify application clusters, kept as simple as
    possible) maps to _build_clusters: weighted-Jaccard entity groups
    become clusters directly, atomicity is enforced, and write-based
    membership is granted — nothing else touches cluster shape after.

A process that never writes (C/U/D) anything — a pure reader — is never a
member of any cluster, no matter how unambiguous its reads are (e.g. it
only ever reads from a single, obvious cluster). Membership comes only
from writing. This matches the classical BSP reference this thesis is
built on: a process shown only reading across several data classes isn't
drawn as belonging to any of the resulting systems. An earlier version of
this pipeline tried to place an unambiguous pure reader into its one
clear best-matching cluster (_place_pure_readers, removed) — dropped once
the reference made clear that even that case should stay unallocated.

Pipeline (see run_bsp):
  1. _compute_entity_owners — resolve which process "owns" each entity
     (the creator; ties broken by C > U > D > R priority). Used for
     reporting (BSPResult.entity_owners); does not drive the clustering.
  2. _reorder_matrix — greedy nearest-neighbour reordering of rows/columns
     by affinity, maximising block-diagonal density.
  3. _build_clusters — group entities by weighted Jaccard similarity, then:
       • Atomic processes: _enforce_atomicity forces all entities they
         WRITE (C/U/D) into the same entity group before clusters are
         even formed (ACID — writes cannot span two systems). Entities
         they only READ may live in other groups. A confirmed override
         (see below) exempts one specific entity from this forced union.
       • Every process (atomic or end_to_end alike, no special-casing)
         becomes a member of every cluster it genuinely writes to — for
         an unexempted atomic process that's always exactly one cluster,
         by construction; for an end_to_end process, or an atomic one
         with a confirmed override, it's naturally however many clusters
         it writes into. This IS what used to be a separate concept
         ("e2e span") — now it's just ordinary membership. A process with
         no writes at all becomes a member of nothing.

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
PROCESSES) and entity_weights (a pair of ENTITIES). Since clusters are
now defined directly by entity groups, entity_weights is the axis with
direct influence on cluster shape (feeds _group_entities). process_weights
no longer has a hand in cluster assembly at all — its role narrows to the
visual matrix reordering and to _detect_pending_conflicts (spotting when
an architect's process-level preference fights the atomicity rule).
Neither axis can silently break an atomic process's forced write
co-location — a conflict between either axis and atomicity only ever
surfaces as a PendingConflict, never applied without confirmation.

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
    unallocated_reads: Dict[str, List[int]] = field(default_factory=dict)  # processo sem cluster → ids de cluster que lê (só relatório)


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
    so it stays in the same scale as the weighted Jaccard used in _build_clusters.

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
    # Lista, não set: a ordem de iteração de um set() de strings muda a
    # cada execução do Python (hash aleatório), o que tornava o desempate
    # do max() abaixo não-determinístico -- a mesma matriz podia reordenar
    # as colunas de forma diferente em cada arranque da app.
    remaining = [i for i in items if i != seed]

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
# A antiga _extract_blocks fazia tudo numa função só. Foi primeiro
# dividida em passos nomeados (threshold adaptativo, tabela de decisão),
# e depois simplificada outra vez: já não há um concurso entre grupos de
# processos e grupos de entidades para decidir quem fica com quem — um
# cluster é diretamente um grupo de entidades, e cada processo entra nos
# clusters onde realmente escreve (ver _assign_process_membership e
# _build_clusters, mais abaixo).

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
    aquele par — o mesmo princípio do bias em _affinity/_group_entities,
    mas já pronta a usar como
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
    Junta itens em grupos, sempre a comparar com a média de um grupo
    inteiro — não com um único membro. Isto evita o problema da
    "corrente": se A e B são parecidos, e B e C são parecidos, A e C podem
    não ter nada a ver um com o outro. Comparar só com um item deixava
    isso passar; comparar com a média do grupo inteiro não deixa.

    Cada item novo é comparado com TODOS os grupos já formados (não só o
    mais recente), e junta-se ao de melhor média, se passar o threshold.
    Sem isto, dois itens com uma relação forte (ex: um entity_weight alto)
    podiam nunca chegar a ser comparados um com o outro, só por haver um
    terceiro item, sem relação nenhuma com nenhum dos dois, colocado entre
    eles pela ordem da matriz — encontrado ao vivo com "Menu" e
    "Inventory Record": ficavam sempre separados quando "Supplier Profile"
    calhava no meio, mesmo com um peso que por si só bastava para os unir.

    bias_fn é opcional — usado por _group_entities para deixar
    entity_weights (os pesos do arquiteto) influenciar o resultado.
    """
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


def _group_entities(df: pd.DataFrame, threshold: float, entity_weights: Dict[tuple, float] = None) -> List[List[str]]:
    """
    Agrupa entidades, com pesos do arquiteto ao nível das entidades
    (entity_weights) a poder influenciar o resultado. Antes disto existir,
    o agrupamento de entidades usava só o sinal estrutural do Jaccard,
    porque o único peso que existia (process_weights) é sobre pares de
    processos e não faz sentido aqui. Agora uma restrição do tipo "estas
    duas entidades têm de ficar em sistemas separados" pode agir
    diretamente neste passo — e, como um cluster nasce diretamente de um
    grupo de entidades (ver _build_clusters), é este o passo com
    influência direta na forma final dos clusters.
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

    NOTA sobre a citação: esta função não vem de nenhum paper da revisão
    de literatura da tese — confirmado ao ler os 21 papers em
    `SLR/Downloaded_Papers/RQ1`. Chegou a estar atribuída a Akkasi,
    Seyyedi & Shams ("Presenting A Method for Benchmarking Application in
    the Enterprise Architecture Planning Process..."), mas essa
    atribuição estava errada — esse paper usa uma fórmula parecida na
    forma (`média ± 2×desvio`), mas para excluir organizações-candidatas
    atípicas de uma lista de parceiros de benchmarking, não para decidir
    se dois processos/entidades devem ficar no mesmo cluster. É um
    threshold estatístico próprio deste projeto, não uma técnica
    importada da literatura.

    A referência certa para o threshold FIXO (0.5) que este substitui,
    opcionalmente, é Lee, H.-S., "Automatic clustering of business
    processes in business systems planning," European Journal of
    Operational Research 114(2), 354-362, 1999 — que testou vários
    valores de threshold para clustering de processos em BSP e escolheu
    k=0.5 experimentalmente, admitindo que "There is no guideline for
    determining suitable k." Foi essa admissão que motivou construir e
    calibrar esta alternativa adaptativa.

    Testado contra as 21 matrizes reais acumuladas em
    `src/resources/initial_matrices/`: a distribuição de similaridade
    entre processos é extremamente enviesada (tipicamente ~87% dos pares
    com jaccard 0 — a maioria dos processos não partilha entidade
    nenhuma), por isso k=1.0 (o valor por omissão aqui) dá um threshold
    muito mais permissivo do que o fixo 0.5 usado até agora — mudava a
    contagem de clusters em 17 das 21 matrizes. k≈2.0 aproxima-se muito
    mais do comportamento já validado (só 7/21 diferem); k=2.5/3.0 satura
    por volta de 5/21, sem melhoria adicional. `adaptive_threshold`
    continua False por omissão em run_bsp — isto é só a calibração do k,
    não uma decisão de ligar isto na app.
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


def _assign_process_membership(df: pd.DataFrame, ent_groups: List[List[str]]) -> List[Cluster]:
    """
    Um cluster passa a ser exatamente um grupo de entidades — nasce
    diretamente de ent_groups, sem nenhum concurso entre processos para o
    "ganhar". A pergunta sobre cada processo é sempre a mesma, sem exceção
    para atómico ou end_to_end: "este processo ESCREVE (C/U/D) nalguma
    entidade deste cluster?" Se sim, é membro — e pode ser membro de
    vários clusters ao mesmo tempo, sem que isso precise de tratamento à
    parte (era isso que process_span + um loop a converter isso em
    membership faziam antes; agora sai correto logo aqui).

    Substitui _pair_groups_to_clusters + _assign_unclaimed_entities, que
    competiam por um único "dono" e depois remendavam quem perdia — e era
    exatamente esse desempate que ficava cego a process_weights. Sem
    concurso, não há desempate para ficar cego a nada.

    Não precisa de saber nada sobre atomicidade: _enforce_atomicity já
    correu antes e já garante que um processo atómico sem override só
    escreve entidades dentro de UM ent_group — só pode ganhar membership
    num cluster. Com um confirmed_override, escreve em dois grupos e fica
    membro dos dois, tal como um end_to_end.
    """
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


def _build_clusters(
    df: pd.DataFrame,
    density_threshold: float = 0.5,
    process_types: Dict[str, str] = None,
    confirmed_overrides: Dict[str, Set[str]] = None,
    entity_weights: Dict[tuple, float] = None,
) -> tuple[List[Cluster], List[Decision]]:
    """
    Orquestra a montagem: agrupa entidades, obriga os processos atómicos a
    ficarem com o que escrevem, e só depois dá a cada processo a sua
    membership real — cada um membro de todos os clusters onde escreve,
    sem concurso nenhum entre processos. Um processo que nunca escreve
    nada (só lê) nunca fica membro de cluster nenhum, mesmo que a leitura
    seja de um único cluster óbvio — membership vem só de escrever.

    Já não recebe process_weights: desde que a montagem deixou de ser um
    concurso, esse peso não tem mais nenhum papel aqui — continua a
    influenciar a reordenação visual da matriz e a deteção de conflitos
    com a atomicidade, só que essas duas coisas não passam por aqui.
    """
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
    Generalizado para também cobrir o caso de um processo atómico com um
    confirmed_override: cria a entidade A e atualiza a entidade B, e o
    arquiteto confirmou que A e B devem ficar em clusters diferentes — o
    processo fica com membership real nos dois (_assign_process_membership
    já trata isto sem caso especial), e esta função reporta esse acesso
    como a mesma dependência cross-system que já existia para end_to_end,
    só que chegada lá por uma decisão confirmada, não por uma classificação.

    Nota: esta função não decide membership nenhuma — só relata o que
    _assign_process_membership já decidiu (recalculando a partir da
    matriz, não olhando para c.processes), filtrado aos processos cujo
    cross-cluster faz sentido reportar (end_to_end, ou atómico com
    override). Um atómico comum nunca aparece aqui, mesmo escrevendo num
    único cluster — não é um "span" digno de nota.

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


def _compute_unallocated_reads(df: pd.DataFrame, clusters: List[Cluster]) -> Dict[str, List[int]]:
    """
    Para cada processo que não é membro de nenhum cluster (um leitor puro
    — ver _assign_process_membership), a lista de clusters cujas entidades
    ele lê. Só para relatório: o processo continua a aceder a dados reais,
    mesmo sem representar nenhum sistema, e isso não deve ficar invisível
    para o arquiteto (nem para quem lê o log).
    """
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


def compute_isa_metrics(
    clusters: List[Cluster],
    crud_matrix: Dict[str, Dict[str, str]],
    process_types: Dict[str, str],
) -> Dict[str, float]:
    """
    Computes ISA quality metrics (Vasconcelos, Sousa & Tribolet, 2008,
    "Enterprise Architecture Analysis: An Information System Evaluation
    Approach") for a given clustering. All 4 metrics reported are literal
    implementations of that paper's formulas (no adapted or invented
    metrics) — see each entry below for how the paper's IS Block / IS
    Service / IS Operation vocabulary maps onto this tool's simpler
    process/entity/cluster model. All metrics in [0, 1]; higher is better.

    RSF   : Average IS blocks per process — 1 = every process lives in exactly one
            system. Matches the paper's formula exactly: #services / Σ(#IS Blocks
            per service) — here "service" = "process", the paper's Business
            Service / IS Service distinction collapses to one concept.
    NAIEF : Entity write responsibility — 1 = single source of truth per entity
            (CUD in one cluster). Matches the paper's formula exactly.
    LCOISF: Cluster cohesion — 1 = an IS Block's own operations mostly agree on
            which entities they touch. Literal implementation of the paper's
            formula: LCOISF = 1 - Σ#LCOISi / (#ISBlock × #ISOperation ×
            #InformationEntity), where #LCOISi is the number of DISTINCT
            entity-sets touched by IS Block i's own operations (here: distinct
            sets of entities touched, across a cluster's member processes,
            grouping processes that touch the exact same set together) and
            #ISOperation is the total process count across the whole ISA. By
            construction (a 3-way product denominator against a numerator that
            grows far more slowly) this saturates close to 1.0 for almost any
            reasonably-organised real system — confirmed against the paper's
            own worked example (0.99 and 1.00 for two genuinely different
            Citizen Card architectures) — so treat differences in the 3rd
            decimal place as meaningful, not differences at a glance.
    CPSMF : Critical/non-critical isolation — 1 = atomic and E2E processes never
            share a cluster. Matches the paper's formula exactly: a cluster
            mixing critical and non-critical processes penalises EVERY process
            in it, not just whichever type is the minority there.
    """
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

    # De onde é que cada entidade é escrita (C/U/D) — não só pelos processos
    # que são membros formais de algum cluster. Um processo puramente leitor
    # nunca é membro de nenhum cluster (ver _assign_process_membership), mas
    # continua a ser um acesso real aos dados; ignorá-lo escondia escritas
    # cruzadas genuínas desta métrica. Para um processo sem cluster, cada
    # entidade que toca ganha uma origem própria e única (o nome do
    # processo) em vez de um id de cluster — continua a contar como mais
    # um sítio de acesso, mesmo sem pertencer formalmente a um sistema.
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

    # LCOISF: literal Vasconcelos et al. formula. Per cluster, group its own
    # member processes by the exact set of entities each one touches (any
    # C/R/U/D); #LCOISi is how many DISTINCT such entity-sets appear among
    # that cluster's processes -- 1 if they all agree on the same footprint
    # (maximally cohesive), higher the more the cluster's processes scatter
    # across unrelated entity combinations.
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

    # CPSMF: isolation of atomic (critical) vs end_to_end (non-critical) processes.
    # Matches Vasconcelos et al. (2008) literally: CPSMF = 1 - (#{critical
    # processes in a cluster that ALSO holds a non-critical process} +
    # #{non-critical processes in a cluster that ALSO holds a critical
    # process}) / #processes. A cluster mixing both types penalises EVERY
    # process in it, not just whichever type is the minority there.
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
    density_threshold : the similarity a pair of processes/entities must
        clear to be grouped together, when adaptive_threshold is False
        (the default). 0.5 follows Lee, H.-S., "Automatic clustering of
        business processes in business systems planning," European
        Journal of Operational Research 114(2), 354-362, 1999 — tested
        several threshold values for BSP process clustering and settled
        on 0.5 experimentally ("There is no guideline for determining
        suitable k").
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

    clusters, decisions = _build_clusters(
        reordered, density_threshold=threshold,
        process_types=process_types, confirmed_overrides=confirmed_overrides,
        entity_weights=entity_weights,
    )                                                                                  # 4. montagem: entidades + atomicidade + membership
    pending_conflicts = _detect_pending_conflicts(
        reordered, decisions, owners, process_weights, process_reasoning, threshold,
        entity_weights=entity_weights, entity_reasoning=entity_reasoning,
    )                                                                                  # 5. atomicity-vs-constraint conflicts

    # process_span é só relatório: cada processo já é membro a sério de
    # todos os clusters onde escreve, desde _assign_process_membership —
    # não é preciso nenhum passo extra a converter isso em membership.
    process_span = _compute_process_span(df, clusters, process_types, confirmed_overrides)

    # Processos que nunca escrevem (leitores puros) não são membros de
    # nenhum cluster — de propósito, ver _assign_process_membership. Isto
    # regista o que continuam a aceder, para não ficarem invisíveis.
    unallocated_reads = _compute_unallocated_reads(df, clusters)

    return BSPResult(clusters=clusters, reordered_matrix=reordered,
                     entity_owners=owners, process_span=process_span,
                     hub_flags=hub_flags, pending_conflicts=pending_conflicts,
                     unallocated_reads=unallocated_reads)

# -*- coding: utf-8 -*-
"""
Código partilhado pelos Testes 2 e 3: leitura da matriz CRUD, dos tipos de
processo e da clusterização de referência, e o Rand Index generalizado para
membership múltiplo.

Importar este módulo também acrescenta `src/` ao sys.path, para que
`from utils.bsp import ...` funcione a partir de qualquer diretório.
"""
import re
import sys
from collections import defaultdict
from pathlib import Path

SRC = Path(__file__).resolve().parent.parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import pandas as pd
from utils.bsp import Cluster, _op_weight, _OP_PRIORITY

LONG_FORMAT_COLS = {"process", "entity", "operation"}


def norm(s):
    return re.sub(r"\s+", " ", str(s)).strip()


def _clean_df(path):
    if str(path).lower().endswith((".xlsx", ".xls")):
        df = pd.read_excel(path, dtype=str)
    else:
        df = pd.read_csv(path, dtype=str, encoding="utf-8-sig")
    df.columns = [str(c).strip() for c in df.columns]
    return df


def _is_long_format(df):
    return LONG_FORMAT_COLS.issubset({c.strip().lower() for c in df.columns})


def load_matrix_wide_excel(path):
    """
    Fallback for a wide CRUD matrix (entities as columns, processes as
    rows) that doesn't already have Process/Entity/Operation columns --
    matches the common EA-course layout: an "ENTITY"/"Entities" header
    row, entity names one row below it, process names starting a few
    columns in, values in the grid below. Auto-detects the header row and
    the first entity column instead of hardcoding a fixed row/column index.
    """
    raw = pd.read_excel(path, header=None)
    header_row = None
    for r in range(min(5, len(raw))):
        if raw.iloc[r].astype(str).str.contains("entit", case=False, na=False).any():
            header_row = r + 1  # entity names are the row right after the "Entities" label
            break
    if header_row is None:
        raise ValueError(
            f"Não encontrei uma linha de cabeçalho de entidades em {path!r} — "
            "este ficheiro não parece ter o formato largo esperado (linha 'Entities' "
            "seguida da linha com os nomes das entidades). Usa antes o formato longo "
            "Process,Entity,Operation."
        )
    header = raw.iloc[header_row]
    first_entity_col = next(i for i, v in enumerate(header) if pd.notna(v))
    entity_cols = [norm(v) if pd.notna(v) else None for v in header[first_entity_col:]]

    matrix = {}
    current_proc = None
    for _, row in raw.iloc[header_row + 1:].iterrows():
        # process name lives in whichever column right before the first entity
        # column has text on this row (handles a couple of nested label columns)
        proc_cell = None
        for c in range(first_entity_col - 1, -1, -1):
            if pd.notna(row[c]):
                proc_cell = row[c]
                break
        if proc_cell is not None:
            current_proc = norm(proc_cell)
        if current_proc is None:
            continue
        for i, ent in enumerate(entity_cols):
            if ent is None:
                continue
            val = row[first_entity_col + i]
            if pd.notna(val):
                op = "".join(sorted(set(str(val).upper().strip())))
                existing = matrix.setdefault(current_proc, {}).get(ent, "")
                matrix[current_proc][ent] = "".join(sorted(set(existing + op)))
    return matrix


def load_matrix(path):
    df = _clean_df(path)
    if _is_long_format(df):
        colmap = {c.strip().lower(): c for c in df.columns}
        matrix = {}
        for _, row in df.iterrows():
            proc = norm(row[colmap["process"]])
            ent = norm(row[colmap["entity"]])
            op = str(row[colmap["operation"]]).strip().upper()
            if not proc or not ent or not op or proc.lower() == "nan":
                continue
            existing = matrix.setdefault(proc, {}).get(ent, "")
            matrix[proc][ent] = "".join(sorted(set(existing + op)))
        return matrix
    if str(path).lower().endswith((".xlsx", ".xls")):
        return load_matrix_wide_excel(path)
    raise ValueError(
        f"{path!r} não tem colunas Process/Entity/Operation e não é um .xlsx "
        "para tentar o formato largo."
    )


def load_process_types(path, all_procs):
    """
    Process,ProcessType (atomic/end_to_end) -- partial list is fine, any
    process not mentioned defaults to "atomic". Empty path -> everyone atomic.
    """
    process_types = {p: "atomic" for p in all_procs}
    if not path:
        return process_types
    df = _clean_df(path)
    colmap = {c.strip().lower(): c for c in df.columns}
    if "process" not in colmap or "processtype" not in colmap:
        raise ValueError(f"{path!r} precisa de colunas Process e ProcessType.")
    for _, row in df.iterrows():
        proc = norm(row[colmap["process"]])
        ptype = norm(row[colmap["processtype"]]).lower()
        if not proc or proc.lower() == "nan" or not ptype or ptype == "nan":
            continue
        process_types[proc] = ptype
    return process_types


def load_reference_clusters(path, matrix):
    """
    Agrupa diretamente por Cluster ID, linha a linha -- um processo com
    linhas em Cluster IDs diferentes fica membro de todos esses clusters
    (mesma semântica que o run_bsp já dá a um processo end_to_end que
    escreve em entidades espalhadas por clusters diferentes). Linhas sem
    Cluster ID são ignoradas, não geram um cluster "nan" nem apagam as
    outras linhas desse processo.
    """
    df = _clean_df(path)
    colmap = {c.strip().lower(): c for c in df.columns}
    cluster_key = next(
        (colmap[k] for k in ("cluster id", "clusterid", "cluster") if k in colmap), None
    )
    if cluster_key is None or "process" not in colmap:
        raise ValueError(
            f"{path!r} precisa de colunas Process e \"Cluster ID\" (ou \"Cluster\")."
        )
    has_entity_col = "entity" in colmap
    has_operation_col = "operation" in colmap

    cluster_procs: dict[str, set] = defaultdict(set)
    cluster_ents: dict[str, set] = defaultdict(set)
    suspicious_writes = []
    for _, row in df.iterrows():
        proc = norm(row[colmap["process"]])
        cid_raw = row[cluster_key]
        if not proc or proc.lower() == "nan":
            continue
        if pd.isna(cid_raw):
            # Uma leitura solta sem cluster é normal (processo não alocado
            # nessa relação). Uma ESCRITA sem cluster é suspeita -- toda
            # escrita devia definir/pertencer a um cluster -- por isso só
            # esta avisamos, não a leitura solta.
            if has_operation_col and has_entity_col and pd.notna(row[colmap["operation"]]):
                op = str(row[colmap["operation"]]).strip().upper()
                if _op_weight(op) > _OP_PRIORITY["R"]:
                    suspicious_writes.append((proc, norm(row[colmap["entity"]]), op))
            continue
        cid = norm(cid_raw)
        is_write = True
        if has_operation_col and pd.notna(row[colmap["operation"]]):
            op = str(row[colmap["operation"]]).strip().upper()
            is_write = _op_weight(op) > _OP_PRIORITY["R"]
        # A plain READ doesn't make a process/entity a member of that
        # cluster -- only a write (C/U/D) does, matching how run_bsp
        # itself treats a pure-read touch (tracked separately as an
        # "unallocated read", never as real cluster membership).
        if is_write:
            cluster_procs[cid].add(proc)
            if has_entity_col and pd.notna(row[colmap["entity"]]):
                cluster_ents[cid].add(norm(row[colmap["entity"]]))

    if suspicious_writes:
        print(f"\nAVISO: {len(suspicious_writes)} linha(s) de ESCRITA (C/U/D) sem Cluster ID em {path!r} -- pode ser um esquecimento no preenchimento da referência (uma leitura solta sem cluster é normal, uma escrita não devia ficar por atribuir):")
        for proc, ent, op in suspicious_writes:
            print(f"  - {proc} / {ent} ({op})")

    mdf = pd.DataFrame(matrix).T
    clusters = []
    for i, cid in enumerate(sorted(cluster_procs), start=1):
        procs = sorted(cluster_procs[cid])
        ents = cluster_ents[cid] if has_entity_col else {
            e for p in procs if p in mdf.index for e in mdf.columns
            if _op_weight(mdf.at[p, e]) > _OP_PRIORITY["R"]
        }
        clusters.append(Cluster(id=i, name=f"Cluster {cid}", processes=procs, entities=sorted(ents)))
    return clusters


def pair_agreement(clusters_a, clusters_b, all_items, attr="processes"):
    """
    Rand Index entre duas clusterizações, generalizado para membership
    múltiplo -- um processo end_to_end (ou com override confirmado) pode
    pertencer a mais que um cluster de cada lado, e o mesmo vale para uma
    entidade que viva em mais que um cluster. "Mesmo cluster" para um par
    (i, j) passa a significar "partilham pelo menos um cluster em comum".

    attr="processes" (default) compara ao nível do processo -- a pergunta
    "o BSP e a referência concordam em quais processos ficam juntos?".
    attr="entities" compara ao nível da entidade -- "concordam em quais
    dados ficam juntos?", que é arguivelmente mais direto, já que o BSP
    agrupa entidades primeiro (Jaccard ponderado) e só depois deriva a
    pertença dos processos a partir de quem escreve o quê.
    """
    membership_a: dict[str, set] = defaultdict(set)
    for c in clusters_a:
        for item in getattr(c, attr):
            membership_a[item].add(c.id)
    membership_b: dict[str, set] = defaultdict(set)
    for c in clusters_b:
        for item in getattr(c, attr):
            membership_b[item].add(c.id)

    items = [p for p in all_items if p in membership_a and p in membership_b]
    n = len(items)
    agree = total = 0
    for i in range(n):
        for j in range(i + 1, n):
            total += 1
            same_a = not membership_a[items[i]].isdisjoint(membership_a[items[j]])
            same_b = not membership_b[items[i]].isdisjoint(membership_b[items[j]])
            if same_a == same_b:
                agree += 1
    return (agree / total if total else None), n

# -*- coding: utf-8 -*-
"""
Shared code for Tests 2 and 3: loading the CRUD matrix, process types and
reference clustering, and the Rand Index. Importing it also adds src/ to sys.path.
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


# Collapses whitespace in a name
def norm(s):
    return re.sub(r"\s+", " ", str(s)).strip()


# Reads a .csv or .xlsx as strings with trimmed column names
def _clean_df(path):
    if str(path).lower().endswith((".xlsx", ".xls")):
        df = pd.read_excel(path, dtype=str)
    else:
        df = pd.read_csv(path, dtype=str, encoding="utf-8-sig")
    df.columns = [str(c).strip() for c in df.columns]
    return df


# True if the table has Process, Entity and Operation columns
def _is_long_format(df):
    return LONG_FORMAT_COLS.issubset({c.strip().lower() for c in df.columns})


# Reads a wide matrix (entities as columns, processes as rows), auto-detecting the header row
def load_matrix_wide_excel(path):
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
        # The process name is the nearest non-empty cell left of the first entity column
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


# Reads the CRUD matrix { process: { entity: ops } } from long format, or wide .xlsx as fallback
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


# Reads Process,ProcessType; processes not listed default to atomic
def load_process_types(path, all_procs):
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


# Reads the reference clustering; a process listed under several Cluster IDs belongs to all of them
def load_reference_clusters(path, matrix):
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
            # A read without a cluster is normal; a write (C/U/D) without one is probably a mistake, so warn
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
        # Only writes (C/U/D) make a process/entity a member of a cluster, same rule as run_bsp
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


# Rand Index between two clusterings, on processes or entities (attr); a pair is 'together' if it shares any cluster
def pair_agreement(clusters_a, clusters_b, all_items, attr="processes"):
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

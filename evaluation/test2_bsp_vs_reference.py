# -*- coding: utf-8 -*-
"""
Teste 2 (genérico): alimenta o bsp.py diretamente com uma matriz CRUD real
(sem passar pela extração da LLM) e compara o clustering do algoritmo —
(a) estrutural puro, (b) depois de 1 iteração de princípios de EA (1
chamada LLM) — contra uma clusterização de referência (tipicamente o To-Be
que os próprios autores do caso definiram).

Pede os ficheiros interativamente, para poder ser reutilizado em qualquer
caso de estudo, não só o DMHU.

Ficheiro 1 — matriz CRUD a enviar para o bsp.py:
  .csv ou .xlsx com colunas Process, Entity, Operation (uma linha por
  operação C/R/U/D; várias linhas por processo/entidade combinam-se numa
  célula, ex. "CU"). Os tipos de processo vêm do ficheiro 3 (opcional).
  Se o ficheiro não tiver essas colunas e for .xlsx, tenta-se o formato
  largo (entidades em colunas, processos em linhas, típico destes
  trabalhos de EA) como alternativa.

Ficheiro 2 — clusterização de referência, para comparar:
  .csv ou .xlsx com colunas Process e Cluster ID (um número — 1, 2, 3...).
  Um processo pode aparecer em várias linhas com Cluster IDs diferentes --
  fica membro de todos esses clusters, tal como o run_bsp permite a um
  processo end_to_end pertencer a mais que um cluster. Linhas com Cluster
  ID em branco são ignoradas (processo não alocado nessa linha), não viram
  um cluster "vazio" nem quebram as outras linhas desse processo.
  Coluna Entity é opcional -- se vier no ficheiro (ex. reaproveitando as
  linhas do ficheiro 1 com uma coluna Cluster ID acrescentada), as
  entidades de cada cluster vêm diretamente dela; senão são inferidas da
  matriz do ficheiro 1 (qualquer entidade tocada por algum processo membro
  desse cluster). Se Entity E Operation vierem os dois, uma linha de
  ESCRITA (C/U/D) sem Cluster ID gera um aviso na consola (provável
  esquecimento) -- uma leitura solta sem cluster é normal e fica em
  silêncio.

Ficheiro 3 (opcional) — tipos de processo:
  .csv ou .xlsx com colunas Process, ProcessType ("atomic" ou
  "end_to_end"). Pode ser um ficheiro à parte, ou reaproveitar um já
  existente com a lista de processos (ex. "group3_processes.csv") só
  acrescentando-lhe a coluna ProcessType. Não precisa de listar todos os
  processos -- só os que não são "atomic" (a maioria dos casos reais é
  "tudo atomic exceto estes N", como o relatório do ALMAFUMO descreve
  explicitamente). Qualquer processo sem valor nessa coluna, ou se o
  ficheiro nem for dado (Enter), fica "atomic" por omissão.
"""
import sys
import io
import os
import asyncio
import json
from collections import defaultdict

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

from common import (
    SRC, load_matrix, load_process_types, load_reference_clusters, pair_agreement,
)
from utils.bsp import run_bsp, compute_isa_metrics, classify_changes
from utils.generator import Generator


def cluster_overlap_table(bsp_clusters, ref_clusters, out):
    """Um processo em mais que um cluster de referência conta para TODOS
    eles (rótulo "A + B"), em vez de só o último encontrado ao iterar."""
    ref_membership: dict[str, list] = defaultdict(list)
    for c in ref_clusters:
        for p in c.processes:
            ref_membership[p].append(c.name)

    out.write("| Cluster BSP | Processos | Cluster(s) de referência (contagem) |\n|---|---|---|\n")
    for c in bsp_clusters:
        counts = {}
        for p in c.processes:
            names = ref_membership.get(p)
            s = " + ".join(sorted(names)) if names else "(não alocado na referência)"
            counts[s] = counts.get(s, 0) + 1
        counts_str = "; ".join(f"{k}: {v}" for k, v in sorted(counts.items(), key=lambda kv: -kv[1]))
        out.write(f"| {c.id} — {c.name} | {len(c.processes)} | {counts_str} |\n")


def export_matrix_xlsx(bsp_result, out_path):
    """Matriz reordenada por cluster, com uma coluna 'Cluster' à frente -- sem
    cor. Um processo em mais que um cluster (end_to_end, ou atomic com
    override confirmado) mostra os dois, não só o último encontrado."""
    proc_cluster: dict[str, list] = defaultdict(list)
    for c in bsp_result.clusters:
        for p in c.processes:
            proc_cluster[p].append(f"{c.id} - {c.name}")
    df = bsp_result.reordered_matrix.copy()
    df.insert(0, "Cluster", [" + ".join(proc_cluster.get(p, [])) for p in df.index])
    df.to_excel(out_path, sheet_name="Matrix")


async def main():
    print("=== Teste 2: BSP direto sobre uma matriz real (sem extração LLM) ===\n")
    matrix_path = input("Ficheiro da matriz CRUD (Process,Entity,Operation — .csv ou .xlsx): ").strip().strip('"')
    ref_path = input("Ficheiro da clusterização de referência (Process, Cluster ID — Entity opcional — .csv ou .xlsx): ").strip().strip('"')
    types_path = input(
        "Ficheiro com os tipos de processo (Process,ProcessType — opcional, Enter para assumir todos atomic): "
    ).strip().strip('"')
    out_dir = input("Pasta onde guardar os resultados: ").strip().strip('"')
    os.makedirs(out_dir, exist_ok=True)

    matrix = load_matrix(matrix_path)
    all_procs = list(matrix.keys())
    all_ents = sorted(set(e for row in matrix.values() for e in row))
    process_types = load_process_types(types_path, all_procs)
    n_e2e = sum(1 for t in process_types.values() if t != "atomic")
    print(f"\nProcessos: {len(all_procs)}, Entidades: {len(all_ents)}")
    if types_path:
        print(f"Tipos de processo: {len(all_procs) - n_e2e} atomic, {n_e2e} não-atomic (de {types_path})")

    ref_clusters = load_reference_clusters(ref_path, matrix)
    ref_metrics = compute_isa_metrics(ref_clusters, matrix, process_types)

    # --- (a) clustering estrutural, sem LLM ---
    structural = run_bsp(matrix, process_types=process_types)
    structural_metrics = compute_isa_metrics(structural.clusters, matrix, process_types)
    ri_structural, n_structural = pair_agreement(structural.clusters, ref_clusters, all_procs)
    ri_structural_ent, n_structural_ent = pair_agreement(structural.clusters, ref_clusters, all_ents, attr="entities")

    # --- (b) 1ª iteração de princípios de EA (1 chamada LLM) ---
    principles_path = SRC / "resources" / "ea_principles.txt"
    with open(principles_path, encoding="utf-8") as f:
        baseline = f.read()

    gen = Generator()
    pw, pr, ew, er = await gen._compute_ea_weights(structural.clusters, baseline, "")

    after_principles = run_bsp(
        matrix, process_types=process_types,
        process_weights=pw, process_reasoning=pr,
        entity_weights=ew, entity_reasoning=er,
    )
    after_metrics = compute_isa_metrics(after_principles.clusters, matrix, process_types)
    ri_after, n_after = pair_agreement(after_principles.clusters, ref_clusters, all_procs)
    ri_after_ent, n_after_ent = pair_agreement(after_principles.clusters, ref_clusters, all_ents, attr="entities")

    changes = classify_changes(structural.clusters, after_principles.clusters)

    out_md = os.path.join(out_dir, "teste2_resultado.md")
    with open(out_md, "w", encoding="utf-8") as out:
        out.write("# Teste 2 — BSP direto sobre uma matriz real\n\n")
        out.write(
            f"Matriz: {matrix_path}, {len(all_procs)} processos, "
            f"{len(set(e for row in matrix.values() for e in row))} entidades — sem passar pela "
            f"extração da LLM. Referência: {ref_path}.\n"
        )

        out.write("\n## (a) Clustering estrutural puro (bsp.py, sem LLM, sem princípios)\n\n")
        out.write(f"nº clusters: {len(structural.clusters)}\n\n")
        out.write(f"Métricas ISA: {json.dumps(structural_metrics)}\n\n")
        out.write(f"Rand Index vs. referência (processos): {ri_structural:.3f} (sobre {n_structural} processos em comum)\n\n")
        out.write(f"Rand Index vs. referência (entidades): {ri_structural_ent:.3f} (sobre {n_structural_ent} entidades em comum)\n\n")
        cluster_overlap_table(structural.clusters, ref_clusters, out)

        out.write("\n## (b) Depois da 1ª iteração de princípios de EA (1 chamada LLM)\n\n")
        out.write(f"nº clusters: {len(after_principles.clusters)}\n\n")
        out.write(f"Métricas ISA: {json.dumps(after_metrics)}\n\n")
        out.write(f"Rand Index vs. referência (processos): {ri_after:.3f} (sobre {n_after} processos em comum)\n\n")
        out.write(f"Rand Index vs. referência (entidades): {ri_after_ent:.3f} (sobre {n_after_ent} entidades em comum)\n\n")
        cluster_overlap_table(after_principles.clusters, ref_clusters, out)

        out.write("\n## O que os princípios de EA mudaram (estrutural -> depois)\n\n")
        out.write(f"{len(changes)} mudança(s):\n\n")
        for ch in changes:
            out.write(f"  - {ch.subject}: cluster {ch.from_cluster} -> {ch.to_cluster} ({ch.change_type}) — {ch.reasoning}\n")

        out.write("\n## Clusterização de referência\n\n")
        for c in ref_clusters:
            out.write(f"- {c.name} ({len(c.processes)} processos)\n")
        out.write(f"\nMétricas ISA da referência: {json.dumps(ref_metrics)}\n")

    export_matrix_xlsx(structural, os.path.join(out_dir, "matriz_estrutural.xlsx"))
    export_matrix_xlsx(after_principles, os.path.join(out_dir, "matriz_apos_principios.xlsx"))

    print("\nGuardado em:", out_md)
    print("Matriz estrutural:", os.path.join(out_dir, "matriz_estrutural.xlsx"))
    print("Matriz após princípios:", os.path.join(out_dir, "matriz_apos_principios.xlsx"))
    print()
    print("Estrutural:", structural_metrics, "Rand Index (processos):", ri_structural, "Rand Index (entidades):", ri_structural_ent)
    print("Depois de EA principles:", after_metrics, "Rand Index (processos):", ri_after, "Rand Index (entidades):", ri_after_ent)
    print("Referência:", ref_metrics)
    print(f"nº clusters -- estrutural: {len(structural.clusters)}, depois: {len(after_principles.clusters)}, referência: {len(ref_clusters)}")


if __name__ == "__main__":
    asyncio.run(main())

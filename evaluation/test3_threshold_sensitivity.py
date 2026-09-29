# -*- coding: utf-8 -*-
"""
Teste 3 (genérico): mesma ideia do test2_bsp_vs_reference.py -- alimenta o bsp.py
diretamente com uma matriz CRUD real, sem passar pela extração da LLM --
mas em vez de aplicar princípios de EA, corre o clustering ESTRUTURAL
puro várias vezes, uma por cada threshold de similaridade testado
(incluindo o threshold adaptativo), e regista como o Rand Index contra a
referência e as 4 métricas de qualidade ISA variam com o threshold.

Os thresholds testados NÃO são números redondos escolhidos às cegas
(ex. 0.2/0.5/0.7) -- a primeira versão deste script fazia isso e deu
gráficos praticamente planos, porque esses valores caíam sempre no mesmo
"vazio" entre duas similaridades reais da matriz, onde threshold nenhum
muda o resultado. Uma segunda versão tentou adivinhar os pontos de
transição a partir das similaridades brutas entre pares de entidades
(bsp.py linha ~611, `_group_entities`), mas isso também não é garantido:
a decisão de juntar uma entidade a um grupo compara-a com a MÉDIA de
similaridade contra todo o grupo (`_average_linkage_group`, bsp.py linha
~584, `if best_avg >= threshold`), não com um par isolado -- essa média
pode cair num valor que não é nenhuma das similaridades brutas. Por
isso, esta versão varre o threshold num grid fino e uniforme (0.01 em
0.01, de 0.0 a 1.0) em vez de tentar adivinhar -- como não há LLM
nenhuma envolvida, cada ponto corre em frações de segundo, por isso não
há razão para arriscar perder uma transição real só para poupar tempo.
Os valores de similaridade brutos continuam a ser impressos no ecrã, só
como contexto informativo. 0.5 (Lee) e o threshold adaptativo entram
sempre no sweep (mesmo que não caiam certinho no grid), marcados com uma
linha vertical nos gráficos, para veres exatamente onde caem.

Não há nenhuma chamada à LLM neste script -- é 100% determinístico, por
isso corre cada threshold uma vez só (sem variação entre corridas).

Motivação: run_bsp usa por omissão density_threshold=0.5, seguindo Lee
(1999), que prova matematicamente que 0.5 maximiza a "net cohesion" e
minimiza o "net coupling" tal como ELE os definiu -- uma otimização
auto-referencial da própria matriz de similaridade, nunca validada
contra uma referência externa e independente (o Lee não tinha nenhuma no
seu exemplo). Este script testa exatamente isso: será que 0.5 é também o
threshold que mais se aproxima do que um arquiteto humano (a referência)
decidiu, e/ou o que dá a melhor arquitetura pelas métricas ISA? Por essa
razão, a métrica de saída aqui é o Rand Index e as métricas ISA -- nunca
a própria fórmula de cohesion/coupling do Lee, que estaria otimizada em
0.5 por construção e não mostraria nada de novo.

Ficheiros de input -- exatamente os mesmos do Teste 2:

Ficheiro 1 -- matriz CRUD a enviar para o bsp.py:
  .csv ou .xlsx com colunas Process, Entity, Operation (uma linha por
  operação C/R/U/D). Se não tiver essas colunas e for .xlsx, tenta-se o
  formato largo (entidades em colunas, processos em linhas) como
  alternativa.

Ficheiro 2 -- clusterização de referência, para comparar:
  .csv ou .xlsx com colunas Process e Cluster ID. Um processo pode
  aparecer em várias linhas com Cluster IDs diferentes -- fica membro de
  todos esses clusters. Coluna Entity é opcional -- se vier no ficheiro,
  as entidades de cada cluster vêm diretamente dela; senão são inferidas
  da matriz do ficheiro 1.

Ficheiro 3 (opcional) -- tipos de processo:
  .csv ou .xlsx com colunas Process, ProcessType ("atomic" ou
  "end_to_end"). Não precisa de listar todos os processos -- só os que
  não são "atomic". Sem este ficheiro, CPSMF fica incorreto para
  qualquer caso com processos end_to_end (fica tudo "atomic" por
  omissão) -- para o DMHU (100% atomic) não faz diferença nenhuma, mas
  para o ALMAFUMO faz, por isso convém dar sempre este ficheiro lá.

Outputs:
  - print no ecrã dos valores de similaridade descobertos, e de nº de
    clusters, métricas ISA e Rand Index (processos e entidades) para
    cada threshold testado.
  - <pasta de saída>/teste3_resultados.csv -- a mesma tabela em bruto.
  - <pasta de saída>/teste3_metricas_isa.png -- RSF/NAIEF/LCOISF/CPSMF
    vs. threshold, em escada (drawstyle="steps-post", já que é
    literalmente uma função em degraus do threshold), com 0.5 (Lee) e o
    adaptativo marcados como linhas verticais.
  - <pasta de saída>/teste3_rand_index.png -- Rand Index (processos e
    entidades) vs. threshold, no mesmo estilo.
  - <pasta de saída>/teste3_n_clusters.png -- nº de clusters vs.
    threshold, para ajudar a explicar as escadas dos outros dois.
"""
import sys
import io
import os

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from common import load_matrix, load_process_types, load_reference_clusters, pair_agreement
from utils.bsp import (
    run_bsp, compute_isa_metrics, _reorder_matrix, _pairwise_scores, _weighted_jaccard,
    derive_threshold,
)


def resolve_adaptive_threshold(matrix, adaptive_k=1.0, adaptive_min_pairs=5, fallback=0.5):
    """Replica o cálculo interno do run_bsp para o adaptive_threshold, só
    para conseguirmos reportar/plotar o valor numérico real que foi usado
    (o BSPResult não o expõe). Nota: o run_bsp deriva este valor a partir
    da distribuição de similaridade entre PROCESSOS (axis="row"), mesmo
    que quem decide os clusters (_group_entities) use a similaridade
    entre ENTIDADES (axis="col") -- replicamos aqui exatamente o que o
    run_bsp faz internamente, não o que "faria mais sentido"."""
    df = pd.DataFrame(matrix).T
    reordered = _reorder_matrix(df)
    proc_scores = _pairwise_scores(
        list(reordered.index),
        lambda a, b: _weighted_jaccard(reordered, a, b, "row"),
    )
    return derive_threshold(proc_scores, k=adaptive_k, min_pairs=adaptive_min_pairs, fallback=fallback)


def discover_entity_breakpoints(matrix):
    """Os únicos thresholds onde a decisão de clustering PODE mudar são os
    valores reais de similaridade ponderada entre pares de ENTIDADES
    (axis="col") -- é esse eixo, não o de processos, que _group_entities
    usa para decidir quem se junta (bsp.py linha ~611). Entre dois valores
    distintos consecutivos, qualquer threshold dá exatamente o mesmo
    resultado, por isso testar em pontos "redondos" como 0.2/0.5/0.7
    arrisca cair sempre na mesma zona morta e não mostrar variação
    nenhuma -- foi o que aconteceu na primeira versão deste script.
    Devolve a lista ordenada de valores distintos >0."""
    df = pd.DataFrame(matrix).T
    reordered = _reorder_matrix(df)
    ent_scores = _pairwise_scores(
        list(reordered.columns),
        lambda a, b: _weighted_jaccard(reordered, a, b, "col"),
    )
    return sorted(set(round(s, 6) for s in ent_scores if s > 0))


def main():
    print("=== Teste 3: sensibilidade do density_threshold do BSP (sem LLM) ===\n")
    matrix_path = input("Ficheiro da matriz CRUD (Process,Entity,Operation -- .csv ou .xlsx): ").strip().strip('"')
    ref_path = input("Ficheiro da clusterização de referência (Process, Cluster ID -- .csv ou .xlsx): ").strip().strip('"')
    types_path = input(
        "Ficheiro com os tipos de processo (Process,ProcessType -- opcional, Enter para assumir todos atomic): "
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
    else:
        print("Sem ficheiro de tipos de processo -- todos assumidos atomic (CPSMF pode ficar incorreto se algum for end_to_end).")

    ref_clusters = load_reference_clusters(ref_path, matrix)
    ref_metrics = compute_isa_metrics(ref_clusters, matrix, process_types)
    print(f"\nReferência: {len(ref_clusters)} clusters, métricas ISA: {ref_metrics}")

    adaptive_value = resolve_adaptive_threshold(matrix)
    print(f"Threshold adaptativo resolvido para esta matriz: {adaptive_value:.4f}")

    breakpoints = discover_entity_breakpoints(matrix)
    print(f"\n{len(breakpoints)} valor(es) distinto(s) de similaridade BRUTA entre entidades (>0): {breakpoints}")
    print("(Só informativo -- o clustering real compara com a MÉDIA de um grupo inteiro, não com estes")
    print(" valores brutos isolados, por isso uma transição pode acontecer num ponto que não está nesta lista.)")

    # Sweep: em vez de tentar adivinhar analiticamente onde estão as
    # transições (o que a lista de breakpoints acima não garante, porque
    # _average_linkage_group compara com a MÉDIA de similaridade contra um
    # grupo inteiro, não com um par isolado -- essa média pode cair em
    # valores que não são nenhuma das similaridades brutas), varremos o
    # threshold num grid fino e uniforme. Como não há nenhuma chamada à
    # LLM, cada ponto corre em frações de segundo, por isso não há razão
    # para arriscar perder uma transição real só para poupar tempo.
    # GRID_STEP mais pequeno = mais fino (mais preciso, mais lento).
    GRID_STEP = 0.01
    sweep = {round(i * GRID_STEP, 6) for i in range(int(1.0 / GRID_STEP) + 1)}
    # garante que os pontos que realmente nos interessam comparar aparecem
    # exatamente, não só o mais próximo do grid:
    sweep.add(0.5)
    sweep.add(round(adaptive_value, 6))
    sweep = sorted(t for t in sweep if 0.0 <= t <= 1.0)

    rows = []
    print(f"\n--- Resultados ao longo de {len(sweep)} thresholds (grid fino de {GRID_STEP}, mais 0.5 e o adaptativo) ---")
    print(f"{'threshold':>9} | {'clusters':>8} | {'RSF':>6} {'NAIEF':>6} {'LCOISF':>6} {'CPSMF':>6} | {'RI proc':>7} {'RI ent':>7}")
    prev_n_clusters = None
    for t in sweep:
        result = run_bsp(matrix, process_types=process_types, density_threshold=t)
        metrics = compute_isa_metrics(result.clusters, matrix, process_types)
        ri_proc, n_proc = pair_agreement(result.clusters, ref_clusters, all_procs, attr="processes")
        ri_ent, n_ent = pair_agreement(result.clusters, ref_clusters, all_ents, attr="entities")

        row = {
            "threshold": t,
            "n_clusters": len(result.clusters),
            "RSF": metrics["RSF"],
            "NAIEF": metrics["NAIEF"],
            "LCOISF": metrics["LCOISF"],
            "CPSMF": metrics["CPSMF"],
            "rand_index_processes": ri_proc,
            "n_processes_compared": n_proc,
            "rand_index_entities": ri_ent,
            "n_entities_compared": n_ent,
        }
        rows.append(row)

        # uma linha compacta por ponto -- não dá para imprimir 4 linhas
        # por cada um dos ~100 pontos do grid e continuar legível. Marca
        # com "<-- muda aqui" sempre que o nº de clusters muda face ao
        # ponto anterior (é aí que está uma transição real), e com o
        # nome (Lee/adaptive) nos dois pontos de referência.
        n_clusters = len(result.clusters)
        changed = " <-- muda aqui" if prev_n_clusters is not None and n_clusters != prev_n_clusters else ""
        tag = ""
        if abs(t - 0.5) < 1e-9:
            tag = " [Lee 0.5]"
        elif abs(t - round(adaptive_value, 6)) < 1e-9:
            tag = " [adaptive]"
        print(f"{t:9.4f} | {n_clusters:8d} | {metrics['RSF']:6.3f} {metrics['NAIEF']:6.3f} "
              f"{metrics['LCOISF']:6.3f} {metrics['CPSMF']:6.3f} | {ri_proc:7.3f} {ri_ent:7.3f}"
              f"{changed}{tag}")
        prev_n_clusters = n_clusters

    # também corre o adaptive_threshold=True "oficial" (deve bater certo
    # com o ponto adaptive_value já incluído no sweep -- serve de confirmação)
    adaptive_result = run_bsp(matrix, process_types=process_types, adaptive_threshold=True)
    adaptive_metrics = compute_isa_metrics(adaptive_result.clusters, matrix, process_types)
    print(f"\nConfirmação -- run_bsp(adaptive_threshold=True) direto: "
          f"{len(adaptive_result.clusters)} clusters, {adaptive_metrics}")

    results_df = pd.DataFrame(rows).sort_values("threshold")
    csv_path = os.path.join(out_dir, "teste3_resultados.csv")
    results_df.to_csv(csv_path, index=False)
    print(f"\nTabela guardada em: {csv_path}")

    # --- estilo geral, para os 3 gráficos ---
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update({
        "font.size": 11,
        "axes.titlesize": 13.5,
        "axes.titleweight": "bold",
        "axes.labelsize": 11,
        "legend.fontsize": 9.5,
        "figure.dpi": 100,
        "grid.alpha": 0.35,
        "grid.linewidth": 0.7,
    })

    def style_ax(ax, ylabel, title):
        ax.set_xlabel("density threshold")
        ax.set_ylabel(ylabel)
        ax.set_title(title, pad=12)
        ax.set_xlim(-0.02, 1.02)
        ax.set_facecolor("#fbfbfd")
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        for spine in ("left", "bottom"):
            ax.spines[spine].set_color("#888888")

    def plot_series(ax, x, y, color, label=None, fill=False):
        """Two layers, so the step shape reads as connected plateaus rather
        than a stack of hard right-angle blocks, without changing the
        underlying (accurate) step shape at all. Base layer: one single
        continuous steps-post line, thin and semi-transparent, with a white
        halo -- this is ONE unbroken path, so it never shows seams, even
        where the real data has many transitions packed close together
        (confirmed with a synthetic dense-oscillation stress test). Top
        layer: a bold, full-colour reinforcement drawn only over each real
        plateau (a run of constant y) -- no halo needed there, since the
        base layer already provides it, so risers stay thin/faint and
        plateaus stand out as bold connected shelves. fill=True adds a
        soft skyline-style fill under the curve -- only safe to use when
        this is the only series on the axes, otherwise overlapping fills
        turn muddy."""
        x = np.asarray(x)
        y = np.asarray(y)
        if fill:
            ax.fill_between(x, y, step="post", color=color, alpha=0.12, zorder=2)
        ax.plot(
            x, y, drawstyle="steps-post", color=color, linewidth=1.3, alpha=0.55,
            solid_joinstyle="round", solid_capstyle="round", zorder=3,
            path_effects=[pe.Stroke(linewidth=2.6, foreground="white"), pe.Normal()],
        )
        change_idx = sorted(set([0] + [i for i in range(1, len(y)) if y[i] != y[i - 1]] + [len(y) - 1]))
        for a, b in zip(change_idx[:-1], change_idx[1:]):
            ax.plot([x[a], x[b]], [y[a], y[a]], color=color, linewidth=3.2,
                     solid_capstyle="round", zorder=4)
        handle, = ax.plot([], [], color=color, linewidth=3.0, label=label, solid_capstyle="round")
        return handle

    def add_reference_lines(ax):
        """Vertical lines for 0.5 (Lee) and the adaptive threshold -- shown
        in the legend like any other series, instead of rotated text
        squeezed against the top of the plot."""
        h1 = ax.axvline(0.5, color="#9a9a9a", linestyle="--", linewidth=1.5,
                         label="Lee (0.5)", zorder=1)
        h2 = ax.axvline(adaptive_value, color="#3a3a3a", linestyle=":", linewidth=1.7,
                         label=f"adaptive ({adaptive_value:.3f})", zorder=1)
        return [h1, h2]

    def mark_intersections(ax, cols, colors, label_above, fmt="{:.3f}"):
        """Marks, with a bold dot and a small value label, exactly where
        each series sits at threshold=0.5 and at the adaptive threshold --
        the two vertical reference lines. Alternates labels above/below
        the point (per column) so they do not overlap each other when two
        series happen to sit close together."""
        for x in (0.5, adaptive_value):
            row = results_df.iloc[(results_df["threshold"] - x).abs().argmin()]
            for i, (col, color) in enumerate(zip(cols, colors)):
                y = row[col]
                ax.scatter([x], [y], s=60, color=color, edgecolor="white",
                           linewidth=1.4, zorder=6)
                above = label_above[i] if label_above else True
                dy = 9 if above else -9
                va = "bottom" if above else "top"
                ax.annotate(fmt.format(y), xy=(x, y), xytext=(6, dy),
                            textcoords="offset points", fontsize=8.3, va=va,
                            color=color, fontweight="bold")

    palette = {
        "RSF": "#2E5EAA", "NAIEF": "#E8871E", "LCOISF": "#1B998B", "CPSMF": "#C1444D",
        "rand_index_processes": "#2E5EAA", "rand_index_entities": "#E8871E",
        "n_clusters": "#7b3f9e",
    }

    # --- chart 1: ISA quality metrics vs. threshold -- one small panel per
    # metric, instead of 4 lines sharing one axes. With 4 step lines
    # overlapping in a small region (0.0-0.2), a single combined axes gets
    # visually busy and any fill turns the overlaps muddy; splitting each
    # metric into its own panel lets every one of them read cleanly, with
    # its own soft fill, while the shared x-axis still makes it easy to
    # compare where each metric sits relative to the two reference lines.
    fig, axes = plt.subplots(2, 2, figsize=(9.5, 6.4), sharex=True)
    cols = ["RSF", "NAIEF", "LCOISF", "CPSMF"]
    for ax, col in zip(axes.flat, cols):
        plot_series(ax, results_df["threshold"], results_df[col], palette[col], fill=True)
        ax.set_title(col, fontsize=12, pad=6)
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(0, 1.08)
        ax.set_facecolor("#fbfbfd")
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        add_reference_lines(ax)
        mark_intersections(ax, [col], [palette[col]], label_above=[True])
    for ax in axes[1]:
        ax.set_xlabel("density threshold")
    for ax in axes[:, 0]:
        ax.set_ylabel("value")
    fig.suptitle("ISA quality metrics vs. threshold", fontweight="bold", fontsize=14, y=0.985)
    fig.text(0.5, 0.94, f"dashed = Lee (0.5)     dotted = adaptive ({adaptive_value:.3f})",
              ha="center", fontsize=9.5, color="#555")
    fig.tight_layout(rect=[0, 0, 1, 0.91])
    isa_png = os.path.join(out_dir, "teste3_metricas_isa.png")
    fig.savefig(isa_png, dpi=160)
    plt.close(fig)
    print(f"Gráfico de métricas ISA guardado em: {isa_png}")

    # --- chart 2: Rand Index vs. threshold ---
    fig, ax = plt.subplots(figsize=(8.5, 5))
    cols = ["rand_index_processes", "rand_index_entities"]
    nice_labels = ["Processes", "Entities"]
    data_handles = [plot_series(ax, results_df["threshold"], results_df[c], palette[c], n)
                     for c, n in zip(cols, nice_labels)]
    style_ax(ax, "Rand Index vs. reference", "Rand Index vs. threshold")
    ax.set_ylim(0, 1.08)
    ref_handles = add_reference_lines(ax)
    mark_intersections(ax, cols, [palette[c] for c in cols], label_above=[True, False])
    ax.legend(handles=data_handles + ref_handles,
              labels=nice_labels + [h.get_label() for h in ref_handles],
              title="Level", frameon=True, framealpha=0.9, loc="best")
    fig.tight_layout()
    ri_png = os.path.join(out_dir, "teste3_rand_index.png")
    fig.savefig(ri_png, dpi=160)
    plt.close(fig)
    print(f"Gráfico de Rand Index guardado em: {ri_png}")

    # --- chart 3: number of clusters vs. threshold, to help explain the other two ---
    fig, ax = plt.subplots(figsize=(8.5, 5))
    plot_series(ax, results_df["threshold"], results_df["n_clusters"], palette["n_clusters"], fill=True)
    style_ax(ax, "number of clusters", "Number of clusters vs. threshold")
    ax.yaxis.get_major_locator().set_params(integer=True)
    ymax = results_df["n_clusters"].max()
    ax.set_ylim(0, ymax + 1)
    ref_handles = add_reference_lines(ax)
    mark_intersections(ax, ["n_clusters"], [palette["n_clusters"]], label_above=[True], fmt="{:.0f}")
    ax.legend(handles=ref_handles, labels=[h.get_label() for h in ref_handles],
              frameon=True, framealpha=0.9, loc="best")
    fig.tight_layout()
    nclusters_png = os.path.join(out_dir, "teste3_n_clusters.png")
    fig.savefig(nclusters_png, dpi=160)
    plt.close(fig)
    print(f"Gráfico de nº de clusters guardado em: {nclusters_png}")


if __name__ == "__main__":
    main()

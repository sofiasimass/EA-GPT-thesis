# -*- coding: utf-8 -*-
"""
Test 3: sensitivity of the BSP density threshold. Runs the structural clustering
(no LLM, fully deterministic) on a real CRUD matrix for thresholds from 0 to 1 in
steps of 0.01, plus Lee's 0.5 and the adaptive threshold, and records the number of
clusters, the ISA metrics and the Rand Index against a reference clustering.

Question: is 0.5 (Lee, 1999) also the threshold closest to the human reference and
the one with the best ISA metrics? A fine grid is used because clusters only change
at specific similarity values that are hard to predict in advance.

Inputs are the same as Test 2. Outputs: teste3_resultados.csv and three charts
(ISA metrics, Rand Index and number of clusters vs. threshold).
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


# Value of the adaptive threshold for this matrix, computed the same way run_bsp does
def resolve_adaptive_threshold(matrix, adaptive_k=1.0, adaptive_min_pairs=5, fallback=0.5):
    df = pd.DataFrame(matrix).T
    reordered = _reorder_matrix(df)
    proc_scores = _pairwise_scores(
        list(reordered.index),
        lambda a, b: _weighted_jaccard(reordered, a, b, "row"),
    )
    return derive_threshold(proc_scores, k=adaptive_k, min_pairs=adaptive_min_pairs, fallback=fallback)


# Distinct raw entity-pair similarities (> 0), printed for context only
def discover_entity_breakpoints(matrix):
    df = pd.DataFrame(matrix).T
    reordered = _reorder_matrix(df)
    ent_scores = _pairwise_scores(
        list(reordered.columns),
        lambda a, b: _weighted_jaccard(reordered, a, b, "col"),
    )
    return sorted(set(round(s, 6) for s in ent_scores if s > 0))


# Runs the threshold sweep and writes the CSV and the three charts
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

    # Fine uniform grid: grouping compares against group AVERAGES, so transitions can't be predicted exactly
    GRID_STEP = 0.01
    sweep = {round(i * GRID_STEP, 6) for i in range(int(1.0 / GRID_STEP) + 1)}
    # Make sure Lee's 0.5 and the adaptive value are tested exactly
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

        # One line per threshold; mark where the number of clusters changes and the two reference points
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

    # Sanity check: run_bsp(adaptive_threshold=True) should match the adaptive point of the sweep
    adaptive_result = run_bsp(matrix, process_types=process_types, adaptive_threshold=True)
    adaptive_metrics = compute_isa_metrics(adaptive_result.clusters, matrix, process_types)
    print(f"\nConfirmação -- run_bsp(adaptive_threshold=True) direto: "
          f"{len(adaptive_result.clusters)} clusters, {adaptive_metrics}")

    results_df = pd.DataFrame(rows).sort_values("threshold")
    csv_path = os.path.join(out_dir, "teste3_resultados.csv")
    results_df.to_csv(csv_path, index=False)
    print(f"\nTabela guardada em: {csv_path}")

    # --- common chart style ---
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

    # Common axis styling
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

    # Step line: thin continuous line plus bold plateaus, optional fill underneath
    def plot_series(ax, x, y, color, label=None, fill=False):
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

    # Vertical lines at Lee's 0.5 and at the adaptive threshold
    def add_reference_lines(ax):
        h1 = ax.axvline(0.5, color="#9a9a9a", linestyle="--", linewidth=1.5,
                         label="Lee (0.5)", zorder=1)
        h2 = ax.axvline(adaptive_value, color="#3a3a3a", linestyle=":", linewidth=1.7,
                         label=f"adaptive ({adaptive_value:.3f})", zorder=1)
        return [h1, h2]

    # Dot and value label where each series crosses the two reference lines
    def mark_intersections(ax, cols, colors, label_above, fmt="{:.3f}"):
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

    # --- chart 1: ISA metrics vs. threshold, one panel per metric ---
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

    # --- chart 3: number of clusters vs. threshold ---
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

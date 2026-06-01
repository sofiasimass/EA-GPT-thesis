
import asyncio
import csv
import json
import os
from utils import Matrix, Entity, Process
from utils import Generator
from utils.bsp import run_bsp, print_bsp_result
import fitz
import pandas as pd
import matplotlib.pyplot as plt
import textwrap

def export_clustered_excel(result, filename):
    df = result.reordered_matrix.fillna("")
    
    # Create a writer object
    with pd.ExcelWriter(filename, engine='openpyxl') as writer:
        df.to_excel(writer, sheet_name='Clustered Matrix')
        workbook  = writer.book
        worksheet = writer.sheets['Clustered Matrix']
        
        # Define a list of soft colors for clusters
        colors = ["E1F5FE", "F1F8E9", "FFFDE7", "F3E5F5", "E8EAF6", "E0F2F1"]
        
        from openpyxl.styles import PatternFill
        
        # Apply highlights based on cluster boundaries
        for i, cluster in enumerate(result.clusters):
            fill = PatternFill(start_color=colors[i % len(colors)], 
                               end_color=colors[i % len(colors)], 
                               fill_type="solid")
            
            # Find the row/column indices in the reordered dataframe
            row_indices = [df.index.get_loc(p) + 2 for p in cluster.processes] # +2 for header/1-index
            col_indices = [df.columns.get_loc(e) + 2 for e in cluster.entities]
            
            for r in row_indices:
                for c in col_indices:
                    worksheet.cell(row=r, column=c).fill = fill

def display_clustered_matrix(result, title="BSP Clustered Matrix"):
    df = result.reordered_matrix.fillna("")
    colors = ["#E1F5FE", "#F1F8E9", "#FFFDE7", "#F3E5F5", "#E8EAF6", "#E0F2F1"]

    cell_colors = [["white"] * len(df.columns) for _ in df.index]
    for i, cluster in enumerate(result.clusters):
        for p in cluster.processes:
            if p in df.index:
                for e in cluster.entities:
                    if e in df.columns:
                        r = list(df.index).index(p)
                        c = list(df.columns).index(e)
                        cell_colors[r][c] = colors[i % len(colors)]

    wrap_width = 12
    wrapped_cols = ["\n".join(textwrap.wrap(col, width=wrap_width)) for col in df.columns]
    max_lines = max(len(w.split("\n")) for w in wrapped_cols)

    fig_w = max(14, len(df.columns) * 0.8 + 4)
    fig_h = max(6, len(df.index) * 0.4 + max_lines * 0.3 + 1)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.set_title(title, fontsize=10, pad=10)
    ax.axis("off")

    table = ax.table(
        cellText=df.values,
        rowLabels=df.index,
        colLabels=wrapped_cols,
        cellColours=cell_colors,
        loc="center"
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.auto_set_column_width(col=list(range(-1, len(df.columns))))

    # Taller header row to fit wrapped text
    header_height = (max_lines * 0.3) / fig_h
    for j in range(len(df.columns)):
        table[0, j].set_height(header_height)

    plt.tight_layout()
    plt.show(block=False)
    plt.pause(0.1)

def read_inputs():

    proc_input = input("Enter processes (comma-separated) OR path to .csv: ").strip()
    if proc_input.lower().endswith('.csv') and os.path.exists(proc_input):
        with open(proc_input, mode='r', encoding='utf-8') as f:
            reader = csv.reader(f)
            processes = [row[0].strip() for row in reader if row and row[0].strip()]
    else:
        processes = [p.strip() for p in proc_input.split(',') if p.strip()]

    info_input = input("Enter information text OR path to file (.txt or .pdf): ").strip()
    
    if os.path.exists(info_input):
        # Handle PDF
        if info_input.lower().endswith('.pdf'):
            text = ""
            with fitz.open(info_input) as doc:
                for page in doc:
                    text += page.get_text()
            context = text
        # Handle TXT
        elif info_input.lower().endswith('.txt'):
            with open(info_input, "r", encoding='utf-8') as f:
                context = f.read()
    else:
        context = info_input

    return processes, context

async def main():

    processes, context = read_inputs()

    ai = Generator()

    print(f"Extracting CRUD matrix ({len(context)} chars, {len(processes)} processes)...")

    final_result = await ai.extract(
            context=context,
            process_list=processes
        )

    with open("full_extraction_log.json", "w") as f:
        json.dump(final_result.model_dump(), f, indent=4)

    print("Full result saved to full_extraction_log.json")

    matrix = Matrix()

    try:
        with open("full_extraction_log.json", "r") as f:
            structured_data = json.load(f)
    except FileNotFoundError:
        print("Error: full_extraction_log.json not found.")
        return
    
    defined_entities = {e["name"] for e in structured_data.get("entities", [])}

    for p in structured_data.get("processes", []):
        matrix.set_process_type(p["name"], p["process_type"])

    for op in structured_data.get("operations", []):
        if op["entity_name"] not in defined_entities:
            print(f"  WARNING: skipping op — entity '{op['entity_name']}' not in defined list")
            continue
        matrix.add_entry(
            p_name=op["process_name"],
            e_name=op["entity_name"],
            operation=op["operation"]
        )

    entity_names_in_ops = {op["entity_name"] for op in structured_data.get("operations", [])}
    for e in structured_data.get("entities", []):
        if e["name"] not in entity_names_in_ops:
            print(f"  WARNING: entity '{e['name']}' has no operations — it won't appear in the matrix")

    output_csv = "final_crud_matrix.csv"
    matrix.export_to_csv(output_csv)
    
    print(f"Matrix generated! {output_csv} contains only the AI-identified data.")

    print("Running initial BSP Clustering...")
    initial_bsp = run_bsp(matrix.matrix, process_types=matrix.process_types)

    export_clustered_excel(initial_bsp, "initial_clustered_matrix.xlsx")
    print("Clustered Excel generated: initial_clustered_matrix.xlsx")
    display_clustered_matrix(initial_bsp, title="Initial BSP Clustering")

    with open("resources/ea_principles.txt", "r") as f:
        baseline_principles = f.read()

    accumulated_user_constraints = ""
    accumulated_weights = {}
    current_bsp = initial_bsp
    iteration = 0
    bsp_iterations = []

    while True:
        answer = input("\nAre you satisfied with the clustering? (yes/no): ").strip().lower()
        if answer in ("yes", "y"):
            break

        constraints = input("Enter your architectural constraints: ").strip()
        if constraints:
            iteration += 1
            accumulated_user_constraints += f"\n\n[Iteration {iteration}]:\n{constraints}"

        print("\nAnalysing constraints and computing EA weights...")
        new_weights = await ai._compute_ea_weights(
            current_bsp.clusters,
            baseline_principles,
            accumulated_user_constraints
        )
        accumulated_weights.update(new_weights)

        current_bsp = run_bsp(matrix.matrix, ea_weights=accumulated_weights, process_types=matrix.process_types)

        out_file = f"clustered_matrix_iter{iteration}.xlsx"
        export_clustered_excel(current_bsp, out_file)
        print(f"Clustered Excel generated: {out_file}")
        display_clustered_matrix(current_bsp, title=f"BSP Iteration {iteration}")

        bsp_iterations.append({
            "iteration": iteration,
            "user_constraint": constraints,
            "ea_weights": {str(k): v for k, v in accumulated_weights.items()},
            "clusters": [c.to_dict() for c in current_bsp.clusters],
            "entity_owners": current_bsp.entity_owners,
        })

    final_bsp = current_bsp

    with open("full_extraction_log.json", "w") as f:
        json.dump({
            "extraction": final_result.model_dump(),
            "initial_bsp": {
                "clusters": [c.to_dict() for c in initial_bsp.clusters],
                "entity_owners": initial_bsp.entity_owners,
            },
            "bsp_iterations": bsp_iterations,
            "final_bsp": {
                "clusters": [c.to_dict() for c in final_bsp.clusters],
                "entity_owners": final_bsp.entity_owners,
                "ea_weights": {str(k): v for k, v in accumulated_weights.items()},
                "user_constraints": accumulated_user_constraints.strip(),
            }
        }, f, indent=4)

    print("Full extraction log updated with clusters!")

if __name__ == "__main__":
    asyncio.run(main())
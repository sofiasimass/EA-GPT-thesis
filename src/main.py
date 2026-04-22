
import asyncio
import csv
import json
import os
from utils import Matrix, Entity, Process
from utils import Generator
from utils.bsp import run_bsp, print_bsp_result
import fitz
import pandas as pd

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

def read_inputs():

    proc_input = input("Enter processes (comma-separated) OR path to .csv: ").strip()
    if proc_input.lower().endswith('.csv') and os.path.exists(proc_input):
        with open(proc_input, mode='r', encoding='utf-8') as f:
            reader = csv.reader(f)
            next(reader) # skip the header
            # take row[2] - process name
            processes = [row[2].strip() for row in reader if row and row[2].strip()]
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

    with open("resources/prompt.txt", "r") as f:
        template_content = f.read()

    processes, context = read_inputs()

    ai = Generator()

    # we only split the extra information / context file because
    # we dont want to loose any information about the business processes
    # and the main instructions in the prompt
    chunks = ai.split(context)

    # Temporary debug inside main()
    print(f"DEBUG: Sending {len(processes)} processes and chunk: {chunks[0][:100]}...")

    final_result = await ai.refine_extraction(
            chunks=chunks,
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
    
    for op in structured_data.get("operations", []):
        matrix.add_entry(
            p_name=op["process_name"], 
            e_name=op["entity_name"], 
            operation=op["operation"]
        )

    output_csv = "final_crud_matrix.csv"
    matrix.export_to_csv(output_csv)
    
    print(f"Matrix generated! {output_csv} contains only the AI-identified data.")

    print("Running initial BSP Clustering...")
    initial_bsp = run_bsp(matrix.matrix)

    export_clustered_excel(initial_bsp, "initial_clustered_matrix.xlsx")
    print("Clustered Excel generated: initial_clustered_matrix.xlsx")

    with open("resources/ea_principles.txt", "r") as f:
        principles = f.read()

    # AI review
    weights = await ai._compute_ea_weights(initial_bsp.clusters, principles)

    # Final architectural run
    final_bsp = run_bsp(matrix.matrix, ea_weights = weights)

    export_clustered_excel(final_bsp, "final_clustered_matrix.xlsx")
    print("Clustered Excel generated: final_clustered_matrix.xlsx")

if __name__ == "__main__":
    asyncio.run(main())
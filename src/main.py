
import asyncio
import csv
import json
import os
from utils import Matrix, Entity, Process
from utils import Generator
import fitz

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

if __name__ == "__main__":
    asyncio.run(main())
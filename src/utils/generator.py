import os
import asyncio
import logging
from dotenv import load_dotenv
from openai import OpenAI
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter


from .matrix import MatrixResult #output schema

class Generator:
    def __init__(self, model_name: str = "gpt-4o"):

        self.llm = ChatOpenAI(model=model_name, temperature=0, request_timeout=120)
        self.structured_output = self.llm.with_structured_output(MatrixResult)

    async def refine_extraction(self, chunks: list, process_list: list):
        # Base instructions from your prompt.txt
        base_instructions = """
        Identify and extract CRUD (Create, Read, Update, Delete) operations 
        from the provided text based on the given list of business processes.
        
        CRITICAL RULES:
        1. Only use process names from the provided Process List.
        2. If a process name in the text is slightly different but clearly refers 
        to one in the list, use the list name.
        """

        # 1. Initial Prompt
        initial_prompt = ChatPromptTemplate.from_template(
            base_instructions + "\n"
            "PROCESS LIST: {process_list}\n"
            "TEXT CHUNK: {raw_document}\n"
            "Extracted CRUD Matrix:"
        )
        
        # 2. Refine Prompt
        refine_prompt = ChatPromptTemplate.from_template(
            base_instructions + "\n"
            "PROCESS LIST: {process_list}\n"
            "------------\n"
            "EXISTING EXTRACTION: {existing_answer}\n"
            "------------\n"
            "NEW TEXT CHUNK: {raw_document}\n"
            "------------\n"
            "INSTRUCTION: Review the NEW TEXT CHUNK. If it contains information that "
            "updates or adds to the EXISTING EXTRACTION, merge it. If it contains new "
            "entities or operations, add them. Output the complete, updated CRUD matrix."
        )

        # Step 1: Initialize
        current_matrix = await (initial_prompt | self.structured_output).ainvoke({
            "process_list": process_list,
            "raw_document": chunks[0]
        })

        # Step 2: Refine Loop
        for i in range(1, len(chunks)):
            chunk = chunks[i]
            success = False
            retries = 3
            
            while not success and retries > 0:
                try:
                    print(f"Refining chunk {i+1}/{len(chunks)}... (Attempt {4-retries})")
                    current_matrix = await (refine_prompt | self.structured_output).ainvoke({
                        "existing_answer": current_matrix.model_dump_json(),
                        "raw_document": chunk,
                        "process_list": process_list
                    })
                    success = True
                except Exception as e:
                    retries -= 1
                    print(f"Connection error on chunk {i+1}: {e}")
                    if retries > 0:
                        print(f"Retrying in 5 seconds...")
                        await asyncio.sleep(5)
                    else:
                        print("Max retries reached. Saving partial progress might be needed.")
                        raise e
                    
        return current_matrix

    def split(self, text: str, chunk_size: int = 12000, chunk_overlap: int = 1000):
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", " ", ""] # Hierarchical splitting
        )
        return splitter.split_text(text)
    
"""
    # this is an alternative to the semaphore approach, using lanchains' abatch that handles concurrency
    async def send_chunks(self, template: str, chunks: list, process_list: list, limit: int = 1):
        prompt = ChatPromptTemplate.from_template(template)
        #Prompt -> LLM -> Output
        chain = prompt | self.structured_output
        
        ## TO DO
        # atm we send the process list in each batch along with a chunk, this is good for the
        # llm to have full context and not forget the processes, however it is unreliable
        # when the process list is long
        batch_inputs = [
            {"process_list": process_list, "extra_information": chunk} 
            for chunk in chunks
        ]
        
        return await chain.abatch(batch_inputs, config={"max_concurrency": limit})
"""

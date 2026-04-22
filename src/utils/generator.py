import os
import asyncio
import logging
from dotenv import load_dotenv
from openai import OpenAI
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
from typing import List

from .bsp import Cluster, EAWeight, EAWeightResult
from .matrix import MatrixResult #output schema

class Generator:
    def __init__(self, model_name: str = "gpt-4o"):

        self.llm = ChatOpenAI(model=model_name, temperature=0, request_timeout=120)
        self.structured_output = self.llm.with_structured_output(MatrixResult)

    async def _compute_ea_weights(self, current_clusters: List[Cluster], ea_principles: str):
        """
        Analyzes the current BSP result and returns a mapping of 
        (item1, item2): weight to refine the next iteration.
        """
        # Create a summary of the draft clusters for the LLM
        # The paper suggests clustering helps understand large software systems 
        cluster_summary = "\n".join([c.summary() for c in current_clusters])
        
        prompt = ChatPromptTemplate.from_template("""
            SYSTEM ROLE:
            You are a Senior Enterprise Architect. You are reviewing a draft clustering of business processes 
            to ensure they align with the company's Enterprise Architecture (EA) Principles.
            
            EA PRINCIPLES: 
            {ea_principles}
            
            CURRENT DRAFT CLUSTERS:
            {cluster_summary}
            
            TASK:
            Review the relationships within and between these clusters.
            - If processes are clustered together but violate principles (e.g., Security vs. Public Access), 
              assign a NEGATIVE weight (down to -0.5) to the pair of processes to push them apart.
            - If processes are split but share a logical business domain, 
              assign a POSITIVE weight (up to 0.5) to the pair of processes to pull them together.
            
            Only provide biases for specific pairs that REQUIRE architectural correction.
            Format your response using the EXACT process names found in the cluster summary.
        """)

        # Use the specialized structured output for weights
        weight_llm = self.llm.with_structured_output(EAWeightResult)
        
        try:
            response = await (prompt | weight_llm).ainvoke({
                "ea_principles": ea_principles,
                "cluster_summary": cluster_summary
            })
            
            # Convert the list of biases into a dictionary for the clustering algorithm
            # Format: {(item_a, item_b): weight}
            weight_map = {}
            for bias in response.biases:
                # We sort the tuple to ensure (A, B) is treated the same as (B, A)
                pair = tuple(sorted([bias.item_a, bias.item_b]))
                weight_map[pair] = bias.weight
                print(f"EA Bias Applied: {pair} -> {bias.weight} ({bias.reasoning})")
            
            return weight_map

        except Exception as e:
            print(f"Error computing EA weights: {e}")
            return {}

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

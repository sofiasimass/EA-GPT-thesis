from pydantic import BaseModel, Field
from typing import List, Optional
import pandas as pd

# --- Structured Output ---

class EntitySchema(BaseModel):
    name: str
    description: str

class EntrySchema(BaseModel):
    process_name: str
    entity_name: str
    operation: str = Field(description="Must be one of C, R, U, or D")

class MatrixResult(BaseModel):
    """The final structured response from the LLM."""
    entities: List[EntitySchema]
    operations: List[EntrySchema]


class Entity:
    def __init__(self, name: str):
        self.name = name

class Process:
    def __init__(self, name: str):
        self.name = name

class Matrix:
    def __init__(self):
        self.matrix = {} # { "Process Name": { "Entity Name": "C" } }
        self.process_objects = {} # all Process objects
        self.entity_objects = {}  # all Entity objects

    def add_entry(self, p_name: str, e_name: str, operation: str):
        # Ensure objects exist (even if just created from the LLM string)
        if p_name not in self.process_objects:
            self.process_objects[p_name] = Process(p_name)
        if e_name not in self.entity_objects:
            self.entity_objects[e_name] = Entity(e_name)

        if p_name not in self.matrix:
            self.matrix[p_name] = {}
        
        self.matrix[p_name][e_name] = operation.upper()

    def export_to_csv(self, filename: str):
        data = []
        for p_name, entities in self.matrix.items():
            for e_name, op in entities.items():
                data.append({
                    'Process': p_name,
                    'Entity': e_name,
                    'Operation': op
                })
        
        df = pd.DataFrame(data)
        if not df.empty:
            # This pivot is the first step toward a BSP matrix
            matrix_df = df.pivot(index='Process', columns='Entity', values='Operation')
            matrix_df.to_csv(filename)
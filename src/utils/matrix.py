from pydantic import BaseModel, Field
from typing import List, Optional
import pandas as pd

# --- Structured Output ---

class EntitySchema(BaseModel):
    name: str = Field(
        description=(
            "The canonical name of a shared data object, e.g. 'Invoice', 'Supplier Profile', "
            "'Purchase Order', 'Product Catalogue', 'Customer Record'. "
            "Must be a concrete data entity — NOT a system name, module name, or process name rephrased. "
            "Use the same name every time this entity appears."
        )
    )
    description: str = Field(
        description="One sentence describing what data this entity holds and why multiple processes share it."
    )

class EntrySchema(BaseModel):
    process_name: str = Field(
        description="Must match EXACTLY one name from the provided process list. No paraphrasing."
    )
    entity_name: str = Field(
        description="Must match EXACTLY one entity name from the entities list in this response."
    )
    operation: str = Field(
        description=(
            "C = this process is the sole originator of the entity. "
            "R = this process reads the entity to function. "
            "U = this process modifies existing records. "
            "D = this process removes records. "
            "Must be exactly one of: C, R, U, D."
        )
    )

class MatrixResult(BaseModel):
    """
    A high-density BSP CRUD matrix.
    DENSITY RULE: every process in the operation list must appear 4 to 6 times
    (i.e. touch 4 to 6 entities). If a process has fewer than 4 operations,
    add more READ entries for reference data it must consult (catalogues, profiles,
    prior records, approval logs). If a process has more than 6, drop the weakest READs.
    ENTITY RULE: produce between 10 and 20 entities — no more. Merge granular or
    overlapping entities into broader shared objects before finalising.
    ORPHAN RULE: every entity you define must appear in at least one operation.
    If you cannot find a natural CRUD operation for an entity, replace it with one
    that does interact with the listed processes.
    """
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
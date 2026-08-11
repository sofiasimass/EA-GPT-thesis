"""
Unit tests for Matrix.add_entry (matrix.py). The LLM emits one operation
entry per operation type, so the same (process, entity) pair can arrive
multiple times -- e.g. a process that fully manages an entity gets
separate C, R, U (and maybe D) entries for it. add_entry must combine
these into one "CRU"-style cell, not overwrite down to the last call.
"""
from utils.matrix import Matrix


def test_multiple_operations_on_the_same_entity_are_combined():
    mat = Matrix()
    mat.add_entry("Manage Clients", "Client", "C")
    mat.add_entry("Manage Clients", "Client", "R")
    mat.add_entry("Manage Clients", "Client", "U")
    mat.add_entry("Manage Clients", "Client", "D")

    cell = mat.matrix["Manage Clients"]["Client"]
    assert set(cell) == {"C", "R", "U", "D"}


def test_repeated_identical_operation_is_not_duplicated():
    mat = Matrix()
    mat.add_entry("Recipe development and testing", "Recipe", "C")
    mat.add_entry("Recipe development and testing", "Recipe", "C")

    assert mat.matrix["Recipe development and testing"]["Recipe"] == "C"


def test_different_entities_are_unaffected_by_combining():
    mat = Matrix()
    mat.add_entry("Purchase order creation and approvals", "Purchase Order", "C")
    mat.add_entry("Purchase order creation and approvals", "Budget", "U")

    assert mat.matrix["Purchase order creation and approvals"]["Purchase Order"] == "C"
    assert mat.matrix["Purchase order creation and approvals"]["Budget"] == "U"

"""Unit tests for the low-level CRUD weighting helpers in bsp.py."""
import math
import pandas as pd

from utils.bsp import _op_weight, _compute_entity_owners


def test_op_weight_single_operations():
    assert _op_weight("C") == 4
    assert _op_weight("U") == 3
    assert _op_weight("D") == 2
    assert _op_weight("R") == 1


def test_op_weight_combined_operations_take_the_highest_priority():
    # C is present, so it wins even though R and U are also present.
    assert _op_weight("CRU") == 4
    # No C here, so U (3) beats R (1).
    assert _op_weight("RU") == 3


def test_op_weight_handles_missing_values():
    assert _op_weight(None) == 0
    assert _op_weight(float("nan")) == 0
    assert _op_weight("") == 0
    assert _op_weight("X") == 0  # unrecognised letter


def test_entity_owner_prefers_create_over_update_and_read():
    matrix = {
        "Create Order": {"Order": "C"},
        "Update Order": {"Order": "U"},
        "View Order":   {"Order": "R"},
    }
    df = pd.DataFrame(matrix).T
    owners = _compute_entity_owners(df)
    assert owners["Order"] == "Create Order"


def test_entity_owner_falls_back_when_no_create_exists():
    matrix = {
        "Update Order": {"Order": "U"},
        "View Order":   {"Order": "R"},
    }
    df = pd.DataFrame(matrix).T
    owners = _compute_entity_owners(df)
    assert owners["Order"] == "Update Order"

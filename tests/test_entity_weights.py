"""
Regression tests for the entity_weights axis (bsp.py) — the second,
independent bias axis alongside process_weights, added because a
process-pair weight alone can't represent "these two entities should be
apart" when a single process creates both (no second process exists to
hold a process_weight against), or when the real conflict is better
expressed at the data level than the process level.
"""
import pandas as pd

from utils.bsp import _group_entities, _enforce_atomicity, _detect_pending_conflicts, _weighted_jaccard


def test_entity_weights_splits_two_structurally_similar_entities():
    # A and B are always touched identically by the same two processes,
    # so weighted Jaccard alone would always group them.
    matrix = {
        "P1": {"A": "C", "B": "C"},
        "P2": {"A": "U", "B": "U"},
    }
    df = pd.DataFrame(matrix).T

    assert _group_entities(df, threshold=0.5) == [["A", "B"]]
    assert _group_entities(df, threshold=0.5, entity_weights={("A", "B"): -0.6}) == [["A"], ["B"]]


def test_entity_weights_joins_two_structurally_unrelated_entities():
    # A and B share no process at all, so Jaccard alone would never group them.
    matrix = {
        "P1": {"A": "C"},
        "P2": {"B": "C"},
    }
    df = pd.DataFrame(matrix).T

    assert _group_entities(df, threshold=0.5) == [["A"], ["B"]]
    assert _group_entities(df, threshold=0.5, entity_weights={("A", "B"): 0.5}) == [["A", "B"]]


def test_entity_weights_still_merges_when_an_unrelated_entity_sits_between_them():
    """
    Real bug found live: _average_linkage_group only compared a new item
    to the most recently opened group, not to every group formed so far.
    "Menu" and "Inventory Record" had a strong entity_weight-boosted bond,
    but with "Supplier Profile" (unrelated to both) sitting between them
    in column order, "Inventory Record" only ever got compared to
    "Supplier Profile"'s group, never to "Menu"'s -- so they never
    merged, no matter how strong the entity_weight. Fixed by comparing
    every new item against every group formed so far, not just the last.

    Here: A and C share no process at all (jaccard alone is 0 for every
    pair), so entity_weight alone has to do the merging work, and B sits
    between them in column order with no relation to either.
    """
    matrix = {
        "P1": {"A": "C"},
        "P2": {"B": "C"},
        "P3": {"C": "C"},
    }
    df = pd.DataFrame(matrix).T

    groups = _group_entities(df, threshold=0.5, entity_weights={("A", "C"): 0.6})

    assert any(set(g) == {"A", "C"} for g in groups)


def test_pending_conflict_from_pure_entity_weight_on_a_self_owned_pair():
    """
    The exact case process_weights alone could never represent: a single
    atomic process creates two entities, and a constraint says they must
    be apart. No second process exists to hold a process_weight against —
    the signal has to come from entity_weights alone.
    """
    matrix = {"P": {"Anchor": "C", "Hub": "C"}}
    df = pd.DataFrame(matrix).T
    process_types = {"P": "atomic"}
    entities = list(df.columns)

    _, decisions = _enforce_atomicity(
        df, entities, [["Anchor", "Hub"]], process_types, confirmed_overrides={}, threshold=0.5,
    )
    entity_owners = {"Anchor": "P", "Hub": "P"}

    conflicts = _detect_pending_conflicts(
        df, decisions, entity_owners, process_weights={}, process_reasoning={}, threshold=0.5,
        entity_weights={("Anchor", "Hub"): -0.6},
    )

    assert len(conflicts) == 1
    assert conflicts[0].entity == "Hub"
    # No "other process" to blame -- both entities are created by the same one.
    assert conflicts[0].conflicting_process == ""


def test_pending_conflict_from_pure_process_weight_still_works_alone():
    """The original mechanism (process_weights only), confirmed still
    working on its own now that entity_weights also exists: two DIFFERENT
    processes each own one of the conflicting entities."""
    df = pd.DataFrame({
        "P":     {"Anchor": "C", "Shared": "U"},
        "Owner": {"Shared": "C"},
    }).T
    process_types = {"P": "atomic", "Owner": "atomic"}
    entities = list(df.columns)

    _, decisions = _enforce_atomicity(
        df, entities, [["Anchor", "Shared"]], process_types, confirmed_overrides={}, threshold=0.5,
    )
    entity_owners = {"Anchor": "P", "Shared": "Owner"}

    conflicts = _detect_pending_conflicts(
        df, decisions, entity_owners,
        process_weights={("Owner", "P"): -0.6}, process_reasoning={}, threshold=0.5,
    )

    assert len(conflicts) == 1
    assert conflicts[0].entity == "Shared"
    assert conflicts[0].conflicting_process == "Owner"


def test_pending_conflict_combines_both_axes_additively():
    """
    Neither axis alone is strong enough to push the counterfactual score
    below threshold, but together they are -- confirms entity_weights and
    process_weights combine (sum), matching _entity_similarity_with_bias's
    "jaccard + weight" formula, not just whichever axis fires first.
    """
    df = pd.DataFrame({
        "P":       {"Anchor": "C", "Shared": "U"},
        "Owner":   {"Shared": "C"},
        "Helper1": {"Anchor": "C", "Shared": "U"},
        "Helper2": {"Anchor": "C", "Shared": "U"},
    }).T
    baseline = _weighted_jaccard(df, "Anchor", "Shared", "col")
    assert baseline > 0.5  # sanity check: this matrix's baseline clears the threshold on its own

    gap = baseline - 0.5
    weak = -(gap * 0.4)    # one axis alone: score stays at 0.5 + 0.6*gap, still above threshold
    strong = -(gap * 0.6)  # both axes together: score drops to 0.5 - 0.2*gap, below threshold

    process_types = {"P": "atomic", "Owner": "atomic", "Helper1": "atomic", "Helper2": "atomic"}
    entities = list(df.columns)
    _, decisions = _enforce_atomicity(
        df, entities, [["Anchor", "Shared"]], process_types, confirmed_overrides={}, threshold=0.5,
    )
    entity_owners = {"Anchor": "P", "Shared": "Owner"}

    alone = _detect_pending_conflicts(
        df, decisions, entity_owners, process_weights={("Owner", "P"): weak}, process_reasoning={}, threshold=0.5,
    )
    combined = _detect_pending_conflicts(
        df, decisions, entity_owners, process_weights={("Owner", "P"): strong}, process_reasoning={}, threshold=0.5,
        entity_weights={("Anchor", "Shared"): strong},
    )

    assert alone == []
    assert len(combined) == 1

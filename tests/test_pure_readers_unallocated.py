"""
Regression tests confirming a pure reader (a process with zero C/U/D
operations anywhere) never becomes a member of any cluster -- membership
comes only from writing.

Replaces test_pure_reader_absorption.py / _place_pure_readers, which used
to place an unambiguous pure reader into its one clear best-matching
cluster. Dropped once the classical BSP reference this thesis is built on
confirmed that a process shown only reading across data classes isn't
drawn as belonging to any resulting system, ambiguous or not -- there's no
"placement" step left in the pipeline at all now, just
_assign_process_membership granting write-based membership.
"""
from utils.bsp import run_bsp


def test_pure_reader_with_one_unambiguous_source_still_gets_no_membership():
    """
    "Read Widget Only" only ever reads Widget, with a single, completely
    unambiguous creator (Create Widget). Under the old _place_pure_readers
    this would have been placed there; now it stays out of every cluster.
    """
    matrix = {
        "Create Widget": {"Widget": "C"},
        "Read Widget Only": {"Widget": "R"},
    }
    process_types = {"Create Widget": "atomic", "Read Widget Only": "atomic"}

    result = run_bsp(matrix, process_types=process_types)

    assert not any("Read Widget Only" in c.processes for c in result.clusters)
    widget_cluster = next(c for c in result.clusters if "Widget" in c.entities)
    assert widget_cluster.processes == ["Create Widget"]


def test_pure_reader_across_several_clusters_gets_no_membership():
    """
    A reader touching entities owned by different clusters -- the more
    obviously ambiguous case -- also gets no membership. Same rule as
    above, just the scenario that originally motivated it.
    """
    matrix = {
        "Create Widget": {"Widget": "C"},
        "Create Gadget": {"Gadget": "C"},
        "Read Both": {"Widget": "R", "Gadget": "R"},
    }
    process_types = {"Create Widget": "atomic", "Create Gadget": "atomic", "Read Both": "atomic"}

    result = run_bsp(matrix, process_types=process_types)

    assert not any("Read Both" in c.processes for c in result.clusters)


def test_writer_process_keeps_its_membership():
    # Sanity check: the rule only excludes processes with zero writes --
    # a process that writes anything is still a normal cluster member.
    matrix = {"Create Widget": {"Widget": "C"}}
    process_types = {"Create Widget": "atomic"}

    result = run_bsp(matrix, process_types=process_types)

    assert any("Create Widget" in c.processes for c in result.clusters)

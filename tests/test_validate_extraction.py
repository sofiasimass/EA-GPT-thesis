"""
Regression tests for _validate_extraction (generator.py) -- the checks
that decide whether an extraction needs a retry. Confirmed live to matter:
a single retry regularly wasn't enough to fully resolve these (motivating
the bounded retry loop in Generator.extract), so these checks firing
correctly and precisely is what that loop's whole safety net depends on.
"""
from utils.generator import _validate_extraction


def _minimal(processes, entities, operations):
    return {
        "processes": [{"name": p, "process_type": "atomic"} for p in processes],
        "entities": [{"name": e, "description": "x"} for e in entities],
        "operations": operations,
    }


def test_density_flags_processes_with_too_few_or_too_many_entries():
    data = _minimal(
        ["Sparse", "JustRight", "Dense"],
        [f"E{i}" for i in range(1, 10)],
        [
            {"process_name": "Sparse", "entity_name": "E1", "operation": "C"},
            *[{"process_name": "JustRight", "entity_name": f"E{i}", "operation": "C"} for i in range(1, 5)],
            *[{"process_name": "Dense", "entity_name": f"E{i}", "operation": "C"} for i in range(1, 10)],
        ],
    )
    issues = _validate_extraction(data, ["Sparse", "JustRight", "Dense"])

    assert any("Sparse" in i and "fewer than the required 4" in i for i in issues)
    assert any("Dense" in i and "more than the allowed 8" in i for i in issues)
    assert not any("JustRight" in i and "CRUD entries" in i for i in issues)


def test_entity_sharing_flags_mostly_private_entities():
    processes = [f"P{i}" for i in range(4)]
    entities = [f"E{i}" for i in range(4)]
    # every entity touched by exactly one process -- no sharing signal at all
    ops = [{"process_name": f"P{i}", "entity_name": f"E{i}", "operation": "C"} for i in range(4)]

    issues = _validate_extraction(_minimal(processes, entities, ops), processes)

    assert any("shared data objects" in i for i in issues)


def test_never_created_flags_a_properly_defined_entity_with_no_create():
    data = _minimal(
        ["Reader"], ["Orphaned"],
        [{"process_name": "Reader", "entity_name": "Orphaned", "operation": "R"}],
    )

    issues = _validate_extraction(data, ["Reader"])

    assert any("needs a Create somewhere" in i and "Orphaned" in i for i in issues)


def test_dangling_reference_is_reported_once_not_twice():
    """
    Real bug found and fixed: a hallucinated entity name (referenced in an
    operation but never defined) used to also be flagged as "needs a
    Create somewhere" alongside the dangling-reference check -- confusing,
    overlapping guidance for what's really one problem, not two.
    """
    data = _minimal(
        ["P1"], ["Real"],
        [
            {"process_name": "P1", "entity_name": "Real", "operation": "C"},
            {"process_name": "P1", "entity_name": "Ghost", "operation": "R"},
        ],
    )

    issues = _validate_extraction(data, ["P1"])

    never_created_hits = [i for i in issues if "needs a Create somewhere" in i and "Ghost" in i]
    dangling_hits = [i for i in issues if "not in the output's entity list" in i]
    assert never_created_hits == []
    assert len(dangling_hits) == 1


def test_zero_operations_flags_a_process_that_vanishes():
    data = _minimal(
        ["Ghost", "Real"], ["E1"],
        [{"process_name": "Real", "entity_name": "E1", "operation": "C"}],
    )

    issues = _validate_extraction(data, ["Ghost", "Real"])

    assert any("Ghost" in i and "zero operations" in i for i in issues)


def test_no_cross_entity_writers_is_flagged():
    """
    Real gap found live (2026-09-11): a 62-process extraction had every
    writing process own exactly one entity and only read the rest -- zero
    cross-entity writes anywhere, despite prompt.txt asking per process
    whether "this process's own transaction genuinely changes a SECOND,
    different entity". Nothing previously checked the aggregate outcome of
    that question, so it went unenforced for the whole extraction.
    """
    processes = [f"P{i}" for i in range(5)]
    entities = [f"E{i}" for i in range(5)]
    # each process CUDs its own single entity, only reads the next one --
    # no process ever writes two different entities.
    ops = []
    for i in range(5):
        ops.append({"process_name": f"P{i}", "entity_name": f"E{i}", "operation": "C"})
        ops.append({"process_name": f"P{i}", "entity_name": f"E{(i + 1) % 5}", "operation": "R"})

    issues = _validate_extraction(_minimal(processes, entities, ops), processes)

    assert any("no process anywhere creates/updates/deletes more than one entity" in i for i in issues)


def test_one_cross_entity_writer_is_enough_to_clear_the_check():
    processes = [f"P{i}" for i in range(5)]
    entities = [f"E{i}" for i in range(5)]
    ops = [{"process_name": f"P{i}", "entity_name": f"E{i}", "operation": "C"} for i in range(5)]
    # P0's own transaction also updates a second, different entity (E1) --
    # the one real cross-entity write the check is looking for.
    ops.append({"process_name": "P0", "entity_name": "E1", "operation": "U"})

    issues = _validate_extraction(_minimal(processes, entities, ops), processes)

    assert not any("creates/updates/deletes more than one entity" in i for i in issues)

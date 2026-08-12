"""
Regression tests for process_span (bsp.py) -- which processes legitimately
touch more than one cluster, and what counts as "touching". Also covers
run_bsp's conversion of process_span into real Cluster.processes
membership, confirmed against the user's own classical-BSP reference
shape: [P1]->[A], [P1,P2]->[B], the same process P1 a genuine member of
both, not just recorded as "visiting" the second one.
"""
from utils.bsp import run_bsp


def test_process_span_only_counts_writes_not_reads():
    """
    Real design correction: reading a reference entity from another
    cluster is completely normal (every process does it) and isn't a
    meaningful "this process spans systems" signal -- only a real write
    (C/U/D) outside a process's own cluster counts.
    """
    matrix = {
        "P1": {"Owned": "C", "ReadOnly": "R"},
        "P2": {"ReadOnly": "C"},
    }
    process_types = {"P1": "end_to_end", "P2": "atomic"}

    result = run_bsp(matrix, process_types=process_types)

    owned_cluster = next(c for c in result.clusters if "Owned" in c.entities)
    assert result.process_span.get("P1") == [owned_cluster.id]


def test_e2e_process_becomes_real_member_of_every_cluster_it_writes_to():
    """
    User-confirmed target shape (matches a textbook BSP worked example):
    [P1]->[A], [P1,P2]->[B] -- same process P1 a real member of both
    clusters, not just metadata saying it "accesses" the second one.
    """
    matrix = {
        "P1": {"A": "C", "B": "CU"},
        "P2": {"B": "RU"},
    }
    process_types = {"P1": "end_to_end", "P2": "end_to_end"}

    result = run_bsp(matrix, process_types=process_types, entity_weights={("A", "B"): -0.5})

    a_cluster = next(c for c in result.clusters if "A" in c.entities)
    b_cluster = next(c for c in result.clusters if "B" in c.entities)
    assert a_cluster.id != b_cluster.id
    assert "P1" in a_cluster.processes
    assert "P1" in b_cluster.processes
    assert set(result.process_span["P1"]) == {a_cluster.id, b_cluster.id}


def test_atomic_process_only_spans_via_explicit_confirmed_override():
    """
    An atomic process with no confirmed_override never spans clusters --
    _enforce_atomicity's hard rule keeps its writes in exactly one
    cluster, completely unaffected by process_span existing at all.
    """
    matrix = {"P1": {"A": "C", "B": "C"}}
    process_types = {"P1": "atomic"}

    result = run_bsp(matrix, process_types=process_types)

    assert len(result.clusters) == 1
    assert result.process_span == {}

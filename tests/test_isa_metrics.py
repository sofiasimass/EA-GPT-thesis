"""
Unit tests for compute_isa_metrics (bsp.py) — the Vasconcelos et al. ISA
quality metrics (RSF, NAIEF, LCOISF, CPSMF, DIIEF) that ground "is this
architecture good" in numbers instead of a read of the output.

Both cases here are hand-computed, not just re-derived from the function
under test — see the comments for the arithmetic.
"""
from utils.bsp import Cluster, compute_isa_metrics


def test_well_scoped_clusters_score_well_except_for_a_shared_read():
    # Two clusters, one process each writes its own entity — a clean split.
    # Create Invoice also *reads* Order (a legitimate cross-reference), which
    # should hurt DIIEF (data spans 2 clusters) without touching NAIEF (Order
    # is still written by exactly one process, in exactly one cluster).
    crud_matrix = {
        "Create Order":   {"Order": "C"},
        "View Order":     {"Order": "R"},
        "Create Invoice": {"Invoice": "C", "Order": "R"},
    }
    process_types = {
        "Create Order": "atomic", "View Order": "atomic", "Create Invoice": "atomic",
    }
    clusters = [
        Cluster(id=1, name="Order", processes=["Create Order", "View Order"], entities=["Order"]),
        Cluster(id=2, name="Invoice", processes=["Create Invoice"], entities=["Invoice", "Order"]),
    ]

    metrics = compute_isa_metrics(clusters, crud_matrix, process_types)

    assert metrics["RSF"] == 1.0     # every process lives in exactly one cluster
    assert metrics["NAIEF"] == 1.0   # each entity still has exactly one writer
    assert metrics["LCOISF"] == 1.0  # every entity pair within a cluster shares a process
    assert metrics["CPSMF"] == 1.0   # nothing mixes atomic with end_to_end
    assert metrics["DIIEF"] == 0.667  # Order is *read* from a second cluster


def test_a_giant_stain_cluster_scores_badly_on_cohesion_and_criticality():
    # One cluster forced to hold two processes that share no data and mix
    # atomic with end_to_end — the "wrong boundary" shape BSP exists to avoid.
    crud_matrix = {
        "Create Order": {"Order": "C"},
        "Segmentation and campaign planning": {"Campaign": "C"},
    }
    process_types = {
        "Create Order": "atomic",
        "Segmentation and campaign planning": "end_to_end",
    }
    clusters = [
        Cluster(
            id=1, name="Mixed",
            processes=["Create Order", "Segmentation and campaign planning"],
            entities=["Order", "Campaign"],
        ),
    ]

    metrics = compute_isa_metrics(clusters, crud_matrix, process_types)

    assert metrics["LCOISF"] == 0.0  # Order and Campaign share no common process
    assert metrics["CPSMF"] == 0.5   # the end_to_end process is stranded in a critical cluster
    assert metrics["DIIEF"] == 1.0   # each entity is still only touched from this one cluster


def test_empty_clusters_return_zeroed_metrics_instead_of_crashing():
    assert compute_isa_metrics([], {}, {}) == {
        "RSF": 0.0, "NAIEF": 0.0, "LCOISF": 0.0, "CPSMF": 0.0, "DIIEF": 0.0,
    }


def test_pure_reader_with_no_cluster_still_lowers_diief():
    """
    Real bug found live: since a pure reader (zero C/U/D) is never a
    cluster member (see _assign_process_membership), the old code only
    ever examined a process's row via `for p in c.processes` -- so a
    reader crossing two clusters was invisible to DIIEF, scoring a
    perfect 1.0 even though the entities are genuinely accessed from
    outside their own cluster. "Pure Reader" here is deliberately left
    out of both clusters' `processes`, matching what the real pipeline
    now produces for it.
    """
    crud_matrix = {
        "Create Order":   {"Order": "C"},
        "Create Invoice": {"Invoice": "C"},
        "Pure Reader":    {"Order": "R", "Invoice": "R"},
    }
    process_types = {p: "atomic" for p in crud_matrix}
    clusters = [
        Cluster(id=1, name="Order", processes=["Create Order"], entities=["Order"]),
        Cluster(id=2, name="Invoice", processes=["Create Invoice"], entities=["Invoice"]),
    ]

    metrics = compute_isa_metrics(clusters, crud_matrix, process_types)

    # Order: touched by cluster 1 (Create Order) + an unaffiliated reader = 2
    # sources. Same for Invoice. DIIEF = 2 entities / (2 + 2) = 0.5.
    assert metrics["DIIEF"] == 0.5


def test_pure_reader_with_no_cluster_still_raises_lcoisf():
    """
    Same root cause, opposite direction: a pure reader can be the ONLY
    real evidence that two entities in someone else's cluster belong
    together, but LCOISF's cohesion check used to only look at formal
    cluster members (`c.processes`), missing it entirely.
    """
    crud_matrix = {
        "Create A":   {"A": "C"},
        "Create B":   {"B": "C"},
        "Reads Both": {"A": "R", "B": "R"},
    }
    process_types = {p: "atomic" for p in crud_matrix}
    clusters = [
        Cluster(id=1, name="A/B", processes=["Create A", "Create B"], entities=["A", "B"]),
    ]

    metrics = compute_isa_metrics(clusters, crud_matrix, process_types)

    assert metrics["LCOISF"] == 1.0

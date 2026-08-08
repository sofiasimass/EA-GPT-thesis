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

"""
Unit tests for compute_isa_metrics (bsp.py) — RSF, NAIEF, LCOISF and CPSMF,
the 4 Vasconcelos et al. ISA quality metrics this tool implements literally,
that ground "is this architecture good" in numbers instead of a read of the
output.

Both cases here are hand-computed, not just re-derived from the function
under test — see the comments for the arithmetic.
"""
from utils.bsp import Cluster, compute_isa_metrics


def test_well_scoped_clusters_score_well_except_for_a_shared_read():
    # Two clusters, one process each writes its own entity — a clean split.
    # Create Invoice also *reads* Order (a legitimate cross-reference), which
    # doesn't touch NAIEF (Order is still written by exactly one process, in
    # exactly one cluster) but does give cluster 2 two processes with
    # DIFFERENT entity footprints (Create Invoice touches {Invoice, Order}),
    # which is exactly what LCOISF's #LCOISi counts.
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
    # Cluster 1's own processes (Create Order, View Order) both touch just
    # {Order} -- 1 distinct entity-set, #LCOIS1=1. Cluster 2 has a single
    # process (Create Invoice) touching {Invoice, Order} -- #LCOIS2=1.
    # LCOISF = 1 - (1+1) / (2 clusters * 3 processes * 2 entities) = 1 - 2/12.
    assert metrics["LCOISF"] == 0.833
    assert metrics["CPSMF"] == 1.0   # nothing mixes atomic with end_to_end


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

    # The cluster's two processes touch two DIFFERENT entity-sets ({Order}
    # and {Campaign}) -- #LCOIS1=2. LCOISF = 1 - 2/(1 cluster*2 procs*2 ents)
    # = 1 - 2/4 = 0.5: this tiny example is small enough that the metric's
    # usual near-1.0 saturation (see compute_isa_metrics docstring) doesn't
    # hide the cohesion problem.
    assert metrics["LCOISF"] == 0.5
    # Vasconcelos et al.: a cluster mixing critical and non-critical processes
    # penalises EVERY process in it, not just the minority type -- both
    # processes here count as mismatched.
    assert metrics["CPSMF"] == 0.0


def test_empty_clusters_return_zeroed_metrics_instead_of_crashing():
    assert compute_isa_metrics([], {}, {}) == {
        "RSF": 0.0, "NAIEF": 0.0, "LCOISF": 0.0, "CPSMF": 0.0,
    }


def test_naief_counts_an_unaffiliated_writer_as_its_own_source():
    """
    A pure reader is never a cluster member (see _assign_process_membership),
    but a process that WRITES an entity without being anyone's formal member
    (edge case, not the common path, but the code must not silently drop it)
    still needs to count as a real, separate write source for NAIEF -- it
    isn't shielded from the "single source of truth" check just because BSP
    didn't assign it to a cluster.
    """
    crud_matrix = {
        "Create Order":     {"Order": "C"},
        "Patch Order Directly": {"Order": "U"},
    }
    process_types = {p: "atomic" for p in crud_matrix}
    clusters = [
        Cluster(id=1, name="Order", processes=["Create Order"], entities=["Order"]),
    ]

    metrics = compute_isa_metrics(clusters, crud_matrix, process_types)

    # Order is written by cluster 1 (Create Order) AND by the unaffiliated
    # "Patch Order Directly" -- 2 distinct write sources for 1 entity.
    assert metrics["NAIEF"] == 0.5

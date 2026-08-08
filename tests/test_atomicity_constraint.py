"""
Regression tests for the atomic-process co-location constraint in
_extract_blocks (bsp.py). This is the mechanism most responsible for BSP's
cluster boundaries: an atomic process's written (C/U/D) entities are forced
into the same cluster, via a transitive union-find.

The two tests below reproduce a real finding from investigating why BSP
clusters didn't become monolithic on one run, but could on another: a single
process that legitimately spans several business domains (CRM, Sales,
Kitchen/Ops, Finance) collapses almost the entire matrix into one cluster if
mislabeled "atomic" instead of "end_to_end" — because the union-find has no
cap on how many entities one process's write footprint can merge together.
"""
from utils.bsp import run_bsp

MATRIX = {
    "Lead intake and qualification": {"Event Lead": "C"},
    "Menu, product and experience design": {
        "Recipe": "C", "Menu Item": "C", "Provenance Catalog": "U",
    },
    "Supplier onboarding and contract setup": {"Supplier Contract": "C"},
    "Purchase order creation and approvals": {"Purchase Order": "C", "Budget": "U"},
    # The bridge process: touches five entities across CRM, Sales, Kitchen/Ops
    # and Finance in a single "atomic" transaction.
    "Events and B2B lifecycle": {
        "Event Lead": "U", "Proposal": "C", "Event Order": "C",
        "Menu Item": "U", "Invoice": "C",
    },
    "Kitchen production and service preparation": {"Prep Log": "C", "Recipe": "U"},
    "Financial management and reconciliation": {
        "Financial Transaction": "C", "Invoice": "U", "Budget": "U",
    },
}

# Every process except the bridge one is unambiguously atomic; only the
# bridge process's classification changes between the two scenarios below.
BASE_PROCESS_TYPES = {
    "Lead intake and qualification": "atomic",
    "Menu, product and experience design": "atomic",
    "Supplier onboarding and contract setup": "atomic",
    "Purchase order creation and approvals": "atomic",
    "Kitchen production and service preparation": "atomic",
    "Financial management and reconciliation": "atomic",
}

ALL_ENTITIES = {e for ops in MATRIX.values() for e in ops}


def test_mislabeling_a_cross_department_process_as_atomic_causes_a_monolith():
    process_types = dict(BASE_PROCESS_TYPES, **{"Events and B2B lifecycle": "atomic"})

    result = run_bsp(MATRIX, process_types=process_types)

    sizes = sorted(len(c.entities) for c in result.clusters)
    assert len(result.clusters) == 2
    # One cluster swallows 11 of the 12 entities; only Supplier Contract
    # survives separately, because its owning process never touches
    # anything the bridge process also touches.
    assert sizes == [1, 11]
    assert sum(sizes) == len(ALL_ENTITIES)


def test_correct_end_to_end_label_keeps_domains_separate():
    process_types = dict(BASE_PROCESS_TYPES, **{"Events and B2B lifecycle": "end_to_end"})

    result = run_bsp(MATRIX, process_types=process_types)

    sizes = sorted(len(c.entities) for c in result.clusters)
    assert len(result.clusters) == 4
    assert sizes == [1, 1, 4, 6]
    assert sum(sizes) == len(ALL_ENTITIES)

    # No single cluster should dominate the matrix the way the monolith did.
    assert max(sizes) < len(ALL_ENTITIES)


def test_atomic_process_never_splits_its_own_written_entities():
    """
    Direct check of the co-location guarantee itself: whatever else happens,
    an atomic process must never end up with its C/U/D entities scattered
    across two different clusters — that would violate the ACID rationale
    the whole constraint exists for.
    """
    process_types = dict(BASE_PROCESS_TYPES, **{"Events and B2B lifecycle": "end_to_end"})
    result = run_bsp(MATRIX, process_types=process_types)

    entity_to_cluster = {
        e: c.id for c in result.clusters for e in c.entities
    }

    for proc, ops in MATRIX.items():
        if process_types.get(proc, "atomic") != "atomic":
            continue
        written = [e for e, op in ops.items() if op.upper() in ("C", "U", "D")]
        cluster_ids = {entity_to_cluster[e] for e in written if e in entity_to_cluster}
        assert len(cluster_ids) <= 1, (
            f"atomic process {proc!r} had its written entities split across "
            f"clusters {cluster_ids}"
        )

"""
Regression tests for cluster-assembly ownership and orphan handling
(bsp.py). _pair_groups_to_clusters must assign a process to the cluster of
what it CREATES, not merely what it updates, even after a confirmed
override has exempted the created entity from atomicity's forced merge.
_assign_unclaimed_entities must never let an orphaned entity vanish, and
must never silently reattach it to a cluster its own entity_weight says it
should be kept apart from.

Note on "anchor" entities: for an atomic process, _enforce_atomicity picks
the first touched entity (in matrix column order, after reordering) as the
"anchor" and never runs decide() on it directly -- only on the entities
touched afterwards. A confirmed_override only has an effect on a
NON-anchor entity; exempting the anchor itself does nothing, since the
anchor is never checked. The exact entity that ends up as anchor is an
implementation detail of _reorder_matrix's greedy ordering, not something
these tests should have to predict from first principles -- the exempted
entity below ("Budget") was picked by running the scenario and observing
which one needed exempting, the same way this bug was found and fixed
live.
"""
from utils.bsp import run_bsp


def test_process_owns_the_cluster_of_what_it_creates_not_merely_updates():
    """
    Real bug found live: "Purchase order creation and approvals" creates
    Purchase Order and updates Budget. After a confirmed split, it used to
    end up owning the Budget cluster (the weaker relationship) because
    _effective_op_weight zeroed its score against Purchase Order (the
    entity it was just split from). Ownership must follow the true CREATE.
    """
    matrix = {
        "Purchase order creation and approvals": {"Purchase Order": "C", "Budget": "U"},
        "Budget tracking and unit P&L management": {"Budget": "U"},
    }
    process_types = {
        "Purchase order creation and approvals": "atomic",
        "Budget tracking and unit P&L management": "atomic",
    }

    result = run_bsp(
        matrix, process_types=process_types,
        confirmed_overrides={"Purchase order creation and approvals": {"Budget"}},
        entity_weights={("Budget", "Purchase Order"): -0.5},
    )

    po_cluster = next(c for c in result.clusters if "Purchase Order" in c.entities)
    budget_cluster = next(c for c in result.clusters if "Budget" in c.entities)
    assert po_cluster.id != budget_cluster.id
    assert "Purchase order creation and approvals" in po_cluster.processes
    # It still genuinely updates Budget -- shows up there too, as an access,
    # not lost, and not silently re-merging the two clusters.
    assert "Purchase order creation and approvals" in budget_cluster.processes
    assert budget_cluster.id in result.process_span.get("Purchase order creation and approvals", [])


def test_orphaned_entity_survives_as_its_own_cluster_instead_of_vanishing():
    """
    Real bug found live: an entity split away by a confirmed override, with
    nothing else in the matrix relating to it, used to get silently
    dropped -- _assign_unclaimed_entities created it with no owning
    process, and _absorb_pure_readers' old survivor filter
    (`if c.processes`) then deleted the whole cluster, entity included.
    """
    matrix = {"Purchase order creation and approvals": {"Purchase Order": "C", "Budget": "U"}}
    process_types = {"Purchase order creation and approvals": "atomic"}

    result = run_bsp(
        matrix, process_types=process_types,
        confirmed_overrides={"Purchase order creation and approvals": {"Budget"}},
        entity_weights={("Budget", "Purchase Order"): -0.5},
    )

    all_entities = {e for c in result.clusters for e in c.entities}
    assert "Budget" in all_entities
    budget_cluster = next(c for c in result.clusters if "Budget" in c.entities)
    # Not just surviving -- genuinely owned by the process that writes it.
    assert "Purchase order creation and approvals" in budget_cluster.processes


def test_assign_unclaimed_entities_vetoes_a_negatively_weighted_reattachment():
    """
    Real bug found live: "Supplier and Procurement Management Process"
    creates Financial Transaction and Supplier Profile with equal raw
    priority. _pair_groups_to_clusters ties and picks one; the loser used
    to get silently reattached to the winner's cluster by
    _assign_unclaimed_entities, completely ignoring a negative
    entity_weight that specifically asked to keep them apart.
    """
    matrix = {
        "Supplier and Procurement Management Process": {
            "Financial Transaction": "C", "Supplier Profile": "CRU",
        },
        "Financial Management and Compliance Process": {"Financial Transaction": "RU"},
    }
    process_types = {
        "Supplier and Procurement Management Process": "end_to_end",
        "Financial Management and Compliance Process": "end_to_end",
    }

    result = run_bsp(
        matrix, process_types=process_types,
        entity_weights={("Financial Transaction", "Supplier Profile"): -0.5},
    )

    ft_cluster = next(c for c in result.clusters if "Financial Transaction" in c.entities)
    sp_cluster = next(c for c in result.clusters if "Supplier Profile" in c.entities)
    assert ft_cluster.id != sp_cluster.id
    # The shared creator follows the textbook BSP shape: owns one, accesses the other.
    assert "Supplier and Procurement Management Process" in sp_cluster.processes
    assert "Supplier and Procurement Management Process" in ft_cluster.processes

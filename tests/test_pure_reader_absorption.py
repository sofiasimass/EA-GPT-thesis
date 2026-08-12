"""
Regression tests for _place_pure_readers (bsp.py): a process that never
writes (C/U/D) anywhere is placed into whichever cluster creates most of
what it reads.

Renamed and rewritten from _absorb_pure_readers as part of the assembly
redesign. The old version MOVED a reader that a competing assignment step
(_pair_groups_to_clusters) had already -- sometimes wrongly -- seeded into
some cluster. That competing step no longer exists: _assign_process_membership
never gives a non-writing process a cluster in the first place, so this
version has nothing to move away from. It PLACES instead of moving.

process_type no longer matters here: an end_to_end pure reader is placed
exactly like an atomic one, because there's no "already assigned" home for
an end_to_end process to be exempted from leaving -- see the redesign notes
in bsp.py's module docstring for the full reasoning.
"""
import pandas as pd

from utils.bsp import Cluster, _place_pure_readers

MATRIX = {
    "Create Widget":     {"Widget": "C"},
    "Create Gadget":     {"Gadget": "C"},
    "Read Widget Report": {"Widget": "R"},
    "Read Both Reports":  {"Widget": "R", "Gadget": "R"},
}
DF = pd.DataFrame(MATRIX).T


def _seed_clusters():
    # Only the writers are seeded -- readers start with no cluster at all,
    # matching how _assign_process_membership actually leaves them.
    return [
        Cluster(id=1, name="Widget", processes=["Create Widget"], entities=["Widget"]),
        Cluster(id=2, name="Gadget", processes=["Create Gadget"], entities=["Gadget"]),
    ]


def test_reader_with_one_clear_creator_is_placed_there():
    clusters = _seed_clusters()
    result = _place_pure_readers(clusters, DF)

    widget_cluster = next(c for c in result if "Create Widget" in c.processes)
    assert "Read Widget Report" in widget_cluster.processes


def test_reader_tied_between_two_creators_is_left_unplaced():
    """
    Reads Widget and Gadget equally, and each is created by a different
    cluster -- no clean winner. Left out of both rather than glued to
    whichever cluster happened to be scanned first: an honest "this
    process doesn't clearly belong anywhere" is more useful than a silent
    guess. This is the same mechanism that stops read-only processes from
    getting glued together just because none of what they read has a
    known creator.
    """
    clusters = _seed_clusters()
    result = _place_pure_readers(clusters, DF)

    for c in result:
        assert "Read Both Reports" not in c.processes


def test_writer_process_is_left_untouched():
    # A process that already has a cluster (because it writes something) is
    # never removed from it or duplicated elsewhere by this step -- only
    # genuinely unplaced processes are considered at all.
    clusters = _seed_clusters()
    result = _place_pure_readers(clusters, DF)

    widget_cluster = next(c for c in result if "Widget" in c.entities)
    gadget_cluster = next(c for c in result if "Gadget" in c.entities)
    assert "Create Widget" in widget_cluster.processes
    assert "Create Widget" not in gadget_cluster.processes


def test_owner_lookup_is_not_corrupted_by_a_writer_in_several_clusters():
    """
    Real bug found while verifying the redesign: a process that creates
    entities in TWO different clusters (normal now -- see
    _assign_process_membership) used to make the owner lookup here
    attribute BOTH entities to whichever of those clusters happened to be
    scanned last, because the old loop searched every column of the
    matrix instead of just the cluster's own entities. A reader of the
    entity that got attributed to the wrong cluster was then placed in
    the wrong home. Fixed by only ever checking a cluster's own entities.
    """
    matrix = {
        "Create Both": {"Widget": "C", "Gizmo": "C"},
        "Read Gizmo":  {"Gizmo": "R"},
    }
    df = pd.DataFrame(matrix).T
    clusters = [
        Cluster(id=1, name="Widget", processes=["Create Both"], entities=["Widget"]),
        Cluster(id=2, name="Gizmo", processes=["Create Both"], entities=["Gizmo"]),
    ]

    result = _place_pure_readers(clusters, df)

    gizmo_cluster = next(c for c in result if "Gizmo" in c.entities)
    widget_cluster = next(c for c in result if "Widget" in c.entities)
    assert "Read Gizmo" in gizmo_cluster.processes
    assert "Read Gizmo" not in widget_cluster.processes

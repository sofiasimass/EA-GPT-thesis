"""
Unit tests for _absorb_pure_readers (bsp.py): a process with zero writes
(pure C/U/D-free reader) should be relocated into whichever cluster owns
(creates) the entity it reads — except end_to_end processes, which are
allowed to read across cluster boundaries and must be left alone.

These call _absorb_pure_readers directly with hand-built clusters rather
than going through the full run_bsp() pipeline, so the starting state is
exact and the test isolates just this one step.
"""
import pandas as pd

from utils.bsp import Cluster, _absorb_pure_readers

MATRIX = {
    "Create Widget":            {"Widget": "C"},
    "Read Widget Report":       {"Widget": "R"},
    "Create Gadget":            {"Gadget": "C"},
    "Read Widget for Marketing": {"Widget": "R"},
}
DF = pd.DataFrame(MATRIX).T

PROCESS_TYPES = {
    "Create Widget": "atomic",
    "Read Widget Report": "atomic",
    "Create Gadget": "atomic",
    "Read Widget for Marketing": "end_to_end",
}


def _seed_clusters():
    # Deliberately seed the pure readers into the *wrong* cluster (not the
    # one that creates what they read), so absorption has real work to do.
    return [
        Cluster(id=1, name="Widget", processes=["Create Widget"], entities=["Widget"]),
        Cluster(id=2, name="Gadget", processes=["Read Widget Report", "Create Gadget", "Read Widget for Marketing"], entities=["Gadget"]),
    ]


def test_atomic_pure_reader_moves_to_its_creators_cluster():
    clusters = _seed_clusters()
    result = _absorb_pure_readers(clusters=clusters, df=DF, process_types=PROCESS_TYPES)

    widget_cluster = next(c for c in result if "Create Widget" in c.processes)
    assert "Read Widget Report" in widget_cluster.processes


def test_end_to_end_pure_reader_is_left_in_place():
    clusters = _seed_clusters()
    result = _absorb_pure_readers(clusters=clusters, df=DF, process_types=PROCESS_TYPES)

    gadget_cluster = next(c for c in result if "Create Gadget" in c.processes)
    assert "Read Widget for Marketing" in gadget_cluster.processes


def test_non_reader_process_is_never_moved():
    clusters = _seed_clusters()
    result = _absorb_pure_readers(clusters=clusters, df=DF, process_types=PROCESS_TYPES)

    # Create Gadget performs a write, so it's not a pure reader and must
    # stay out of the Widget-owning cluster regardless of what else moves.
    widget_cluster = next(c for c in result if "Create Widget" in c.processes)
    assert "Create Gadget" not in widget_cluster.processes

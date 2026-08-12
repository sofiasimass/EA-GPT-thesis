"""
Regression test for _greedy_reorder (bsp.py) -- the greedy nearest-neighbour
column/row reordering that runs before clustering.
"""
from utils.bsp import _greedy_reorder


def test_greedy_reorder_breaks_ties_by_input_order_not_by_hash_randomization():
    """
    Real bug found live: `remaining` used to be a set(), whose iteration
    order depends on Python's per-process string hash randomization. When
    two candidates tied on affinity, which one got picked next used to
    vary from run to run of the app -- same matrix, same weights,
    different column order each time the server restarted. That in turn
    made _group_entities's outcome non-deterministic, since it depends on
    column order. Fixed by using a list, whose iteration order is stable
    and matches the original item order.

    Here B and C are genuinely tied in affinity to A -- the correct,
    deterministic pick is B, since it comes first in the input list.
    """
    items = ["A", "B", "C", "D"]

    def affinity(a, b):
        pairs = {
            frozenset(("A", "B")): 1.0,
            frozenset(("A", "C")): 1.0,
            frozenset(("A", "D")): 0.0,
            frozenset(("B", "C")): 0.0,
            frozenset(("B", "D")): 0.0,
            frozenset(("C", "D")): 0.0,
        }
        return pairs[frozenset((a, b))]

    assert _greedy_reorder(items, affinity) == ["A", "B", "C", "D"]

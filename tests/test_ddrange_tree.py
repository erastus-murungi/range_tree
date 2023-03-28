import numpy as np
import pytest

from dd_range_tree import DDRangeTree
from rangetree import Interval, Orthotope
from utils import brute_range_search


@pytest.mark.parametrize(
    "n_iters, n_dims, n_points, rand_num_lim, ray",
    [
        (100, 3, 100, 1000, (0, 100)),
        (100, 3, 100, 1000, (0, 500)),
        (500, 3, 1000, 1000, (0, 1000)),
        (100, 4, 100, 1000, (0, 100)),
        (100, 4, 100, 1000, (0, 500)),
        (500, 4, 1000, 1000, (0, 1000)),
        (100, 6, 100, 1000, (0, 100)),
        (100, 6, 100, 1000, (0, 500)),
        (500, 6, 1000, 1000, (0, 1000)),
    ],
)
def test_range_search(n_iters, n_dims, n_points, rand_num_lim, ray):
    orthotope = Orthotope([Interval(*ray) for _ in range(n_dims)])
    for _ in range(n_iters):
        points = np.random.randint(0, rand_num_lim, (n_points, n_dims))
        range2d_tree = DDRangeTree.construct(points)
        query_iter = range2d_tree.query(orthotope)

        actual_in_range = list(sorted([tuple(map(int, elem)) for elem in query_iter]))
        expected_in_range = list(sorted(brute_range_search(points, orthotope)))

        assert actual_in_range == expected_in_range

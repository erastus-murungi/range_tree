import numpy as np
import pytest

from kdtree import KDTree, Leaf
from rangetree import Interval, Orthotope
from utils import (
    brute_k_nearest_neighbors,
    brute_nearest_neighbor,
    brute_range_search,
    l2_norm,
)


@pytest.mark.parametrize(
    "n_iters, n_points, n_dims, rand_num_lim",
    [(10, 100, 2, 1000), (10, 100, 3, 1000), (5, 1000, 4, 100)],
)
def test_random_inserts_followed_by_deletions(n_iters, n_points, n_dims, rand_num_lim):
    for _ in range(n_iters):
        kd_tree = KDTree(n_dims=n_dims)
        points = np.random.randint(0, rand_num_lim, (n_points, n_dims))
        for point in points:
            kd_tree.insert(point)
            assert point in kd_tree
        for point in points:
            kd_tree.remove(point)
        assert isinstance(kd_tree._root, Leaf) and kd_tree._root.data.size == 0


@pytest.mark.parametrize(
    "n_iters, n_points, n_dims, rand_num_lim",
    [(100, 100, 2, 1000), (100, 100, 3, 1000), (500, 1000, 4, 1000)],
)
def test_nearest_neighbor(n_iters, n_points, n_dims, rand_num_lim):
    for _ in range(n_iters):
        points = np.random.randint(0, rand_num_lim, (n_points, n_dims))
        reference_point = np.random.randint(0, rand_num_lim, (n_dims,))
        kd_tree = KDTree(data_points=points)
        actual_nn = kd_tree.nearest_neighbor(reference_point).point
        expected_nn = brute_nearest_neighbor(points, reference_point, l2_norm)
        assert l2_norm(actual_nn, reference_point) == l2_norm(
            expected_nn, reference_point
        )


@pytest.mark.parametrize(
    "n_iters, n_points, n_dims, rand_num_lim, k",
    [(100, 100, 2, 1000, 5), (100, 100, 3, 1000, 7), (500, 1000, 4, 1000, 10)],
)
def test_k_nearest_neighbors(n_iters, n_points, n_dims, rand_num_lim, k):
    for _ in range(n_iters):
        points = np.random.randint(0, rand_num_lim, (n_points, n_dims))
        reference_point = np.random.randint(0, rand_num_lim, (n_dims,))
        kd_tree = KDTree(data_points=points)
        query_results = kd_tree.k_nearest_neighbors(reference_point, k)
        actual_distances = [dist for _, dist in query_results]
        expected_distances = [
            dist
            for _, dist in brute_k_nearest_neighbors(
                points, reference_point, k, l2_norm
            )
        ]
        assert actual_distances == expected_distances


@pytest.mark.parametrize(
    "n_iters, n_points, n_dims, rand_num_lim, ray",
    [
        (100, 100, 2, 1000, (0, 100)),
        (100, 100, 3, 1000, (0, 500)),
        (500, 1000, 4, 1000, (0, 1000)),
    ],
)
def test_range_search(n_iters, n_points, n_dims, rand_num_lim, ray):
    orthotope = Orthotope([Interval(*ray) for _ in range(n_dims)])
    for _ in range(n_iters):
        points = np.random.randint(0, rand_num_lim, (n_points, n_dims))
        kd_tree = KDTree(data_points=points)
        query_iter = kd_tree.range_search(orthotope)

        actual_in_range = list(sorted([tuple(map(int, elem)) for elem in query_iter]))
        expected_in_range = list(sorted(brute_range_search(points, orthotope)))

        assert actual_in_range == expected_in_range

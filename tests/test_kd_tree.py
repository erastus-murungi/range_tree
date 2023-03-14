from bisect import insort
from operator import attrgetter
from sys import maxsize

import numpy as np
import pytest

from kdtree import KDTree, Leaf, NNResult, l2_norm
from rangetree import Interval, Orthotope


def brute_nearest_neighbor(coords, query_point, distance_function):
    # naive nearest neighbor
    best_dist, best_point = maxsize, None
    for coord in coords:
        dist = distance_function(coord, query_point)
        if dist < best_dist:
            best_dist, best_point = dist, coord
    return best_point


def brute_k_nearest_neighbors(coords, query_point, k, distance_function):
    """Simple kNN for benchmarking"""
    bpq = []
    for coord in coords:
        dist = distance_function(coord, query_point)
        if len(bpq) < k or dist < bpq[-1].distance:
            insort(bpq, NNResult(coord, dist), key=attrgetter("distance"))
            if len(bpq) > k:
                bpq.pop()
    return bpq


def brute_range_search(coords, orthotope):
    for coord in coords:
        if all(x in interval for x, interval in zip(coord, orthotope)):
            yield tuple(coord)


@pytest.mark.parametrize(
    "n_iters, n_points, n_dims, rand_num_lim",
    [(10, 100, 2, 1000), (10, 100, 3, 1000), (5, 1000, 4, 100)],
)
def test_random_inserts_followed_by_deletions(n_iters, n_points, n_dims, rand_num_lim):
    for _ in range(n_iters):
        kd_tree = KDTree(n_dims=n_dims)
        ps = np.random.randint(0, rand_num_lim, (n_points, n_dims))
        for p in ps:
            kd_tree.insert(p)
            assert p in kd_tree
        for p in ps:
            kd_tree.remove(p)
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
        actual = kd_tree.nearest_neighbor(reference_point).point
        expected = brute_nearest_neighbor(points, reference_point, l2_norm)
        assert l2_norm(actual, reference_point) == l2_norm(expected, reference_point)


@pytest.mark.parametrize(
    "n_iters, n_points, n_dims, rand_num_lim, k",
    [(100, 100, 2, 1000, 5), (100, 100, 3, 1000, 7), (500, 1000, 4, 1000, 10)],
)
def test_k_nearest_neighbors(n_iters, n_points, n_dims, rand_num_lim, k):
    for _ in range(n_iters):
        points = np.random.randint(0, rand_num_lim, (n_points, n_dims))
        reference_point = np.random.randint(0, rand_num_lim, (n_dims,))
        kd_tree = KDTree(data_points=points)
        result = kd_tree.k_nearest_neighbors(reference_point, k)
        res_n = [dist for _, dist in result]
        res_m = [
            dist
            for _, dist in brute_k_nearest_neighbors(
                points, reference_point, k, l2_norm
            )
        ]
        assert res_n == res_m


@pytest.mark.parametrize(
    "n_iters, n_points, n_dims, rand_num_lim, ray",
    [
        (100, 100, 2, 1000, (0, 100)),
        (100, 100, 3, 1000, (0, 500)),
        (500, 1000, 4, 1000, (0, 1000)),
    ],
)
def test_range_search(n_iters, n_points, n_dims, rand_num_lim, ray):
    for _ in range(n_iters):
        points = np.random.randint(0, rand_num_lim, (n_points, n_dims))
        kd_tree = KDTree(data_points=points)
        orthotope = Orthotope([Interval(*ray) for _ in range(n_dims)])
        result = kd_tree.range_search(orthotope)

        res_n = list(sorted([tuple(map(int, elem)) for elem in result]))
        res_m = list(sorted(brute_range_search(points, orthotope)))

        assert res_n == res_m

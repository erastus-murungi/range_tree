import numpy as np
import pytest

from utils import brute_k_nearest_neighbors, l2_norm
from vptree import VPTree


@pytest.mark.parametrize(
    "n_iters, n_points, n_dims, rand_num_lim, k",
    [(100, 100, 2, 1000, 5), (100, 100, 3, 1000, 7), (500, 1000, 4, 1000, 10)],
)
def test_k_nearest_neighbors(n_iters, n_points, n_dims, rand_num_lim, k):
    for _ in range(n_iters):
        points = np.random.randint(0, rand_num_lim, (n_points, n_dims))
        reference_point = np.random.randint(0, rand_num_lim, (n_dims,))
        vp_tree = VPTree(points)
        results, _ = vp_tree.k_nearest_neighbors(reference_point, k)
        actual_distances = [dist for _, dist in results]
        expected_distances = [
            dist
            for _, dist in brute_k_nearest_neighbors(
                points, reference_point, k, l2_norm
            )
        ]
        assert actual_distances == expected_distances

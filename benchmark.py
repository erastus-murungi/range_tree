from kdtree import KDTree, brute_k_nearest_neighbors
from vp_tree import VPTree
from time import perf_counter
from pympler import asizeof
from random import randint
from copy import deepcopy
from scipy.spatial.distance import minkowski


if __name__ == '__main__':
    LIM = 100_000
    NUM_COORDS = 100000
    DIM = 5
    k = 10
    coords = [tuple(randint(0, LIM) for _ in range(DIM)) for _ in range(NUM_COORDS)]
    coords3 = deepcopy(coords)
    coords1 = deepcopy(coords)
    coords2 = deepcopy(coords)
    query_point = tuple(randint(0, LIM) for _ in range(DIM))

    t1 = perf_counter()
    kd = KDTree(coords3)
    print(f"k-d tree built in {(perf_counter() - t1):.2f} seconds & used"
          f" {asizeof.asizeof(kd) / (1 << 20) :.2f} MB for {NUM_COORDS} {DIM}-D points.")
    t2 = perf_counter()
    vp = VPTree(coords1, minkowski)
    print(f"v-p tree built in {(perf_counter() - t2):.2f} seconds & used"
          f" {asizeof.asizeof(vp) /(1 << 20) :.2f} MB for {NUM_COORDS} {DIM}-D points.")

    t3 = perf_counter()
    q1 = kd.k_nearest_neighbors(query_point, k)
    print(f"kd kNN query ran in {(perf_counter() - t3):.2f} seconds for {k} nearest neighbors")

    t4 = perf_counter()
    q2 = vp.k_nearest_neighbors(query_point, k)
    print(f"vp kNN query ran in {(perf_counter() - t4):.2f} seconds for {k} nearest neighbors")

    t5 = perf_counter()
    q3 = brute_k_nearest_neighbors(coords2, query_point, k, minkowski)
    print(f"brute kNN query ran in {(perf_counter() - t5):.2f} seconds for {k} nearest neighbors")

    print(list(q1))
    print(list(q2[0]))
    print(list(q3))

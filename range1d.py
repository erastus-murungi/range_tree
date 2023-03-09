from sys import maxsize
from typing import Iterable

import numpy as np
from more_itertools import only

from rangetree import OUT_OF_BOUNDS, Leaf, Node, RangeTree, Split


class RangeTree1D(RangeTree):
    """A 1D Range Tree."""

    def __init__(self, values: np.ndarray, axis: int = 0):
        self.axis = axis
        super().__init__(self.build_range_tree(values))

    def build_range_tree(self, values: np.ndarray) -> Node:
        """Build a 1D Range Tree from the bottom up and returns the root, and the nodes on the same level.
        This is just for indexing.
        It is possible to augment the structure to store any information.
        """
        # O(n log n) because of sorting

        leaves = list(map(Leaf, sorted(values, key=lambda row: row[self.axis])))
        # n + n/2 + n/4 + n/8 + ... + 1 ≅ 2n  (Geometric summation) = O(n)
        while (n_leaves := len(leaves)) > 1:
            nodes = [
                Split(
                    self.split_value(leaves[i - 1], self.axis), leaves[i - 1], leaves[i]
                )
                for i in range(1, n_leaves, 2)
            ]
            if n_leaves & 1:
                nodes.append(
                    Split(
                        self.split_value(leaves[n_leaves - 1], self.axis),
                        leaves[n_leaves - 1],
                        OUT_OF_BOUNDS,
                    )
                )
            leaves = nodes

        # Total running time is: O(n log n)
        return only(
            leaves,
            too_long="Expected to only have one node after binary tree construction",
        )

    def query_range_tree1d(self, min_value, max_value) -> Iterable:
        """Queries a 1D Range Tree.
        Let P be a set of n points in 1-D space. The set P
        can be stored in a balanced binary search tree, which uses O(n) storage and
        has O(n log n) construction time, such that the points in a query range can be
        reported in time O(k + log n), where k is the number of reported points.
        """

        if min_value > max_value:
            min_value, max_value = max_value, min_value

        v_split = self.find_split_node(min_value, max_value)
        if isinstance(v_split, Leaf):
            # check if the point in v_split
            if min_value <= v_split.point[self.axis] < max_value:  # inclusive version
                yield from self.report_leaves(v_split)
        else:
            v = v_split.left
            while not isinstance(v, Leaf):
                if v.split_value >= min_value:
                    # report right subtree
                    yield from self.report_leaves(v.right)
                    v = v.left
                else:
                    v = v.right
            # v is now a leaf
            if min_value <= v.point[self.axis] < max_value:
                yield v.point
            # now we follow right side
            v = v_split.right
            while not isinstance(v, Leaf):
                if v.split_value < max_value:
                    # report left subtree
                    yield from self.report_leaves(v.left)
                    # it is possible to traverse to an external node
                    if v.right is OUT_OF_BOUNDS:
                        return
                    v = v.right
                else:
                    v = v.left
            # check whether this point should be included too
            if min_value <= v.point[self.axis] < max_value:
                yield from self.report_leaves(v)

    def __getitem__(self, item: slice):
        """Assumes item is a slice object.
        To search for a specific value:
        Use that value in both endpoints. eg to search for 5, query [5:5].
        Returns the items in the range.
        """
        assert isinstance(item, slice), print(item)

        start, stop = item.start, item.stop
        if start is None:
            start = 0
        if stop is None:
            stop = maxsize
        if start > stop:
            raise IndexError("make sure start <= stop")

        return self.query_range_tree1d(start, stop)


def brute(points, x1, x2):
    for point in points:
        if x1 <= point < x2:
            yield point


if __name__ == "__main__":
    from random import randint

    num_points = 100
    x1, x2 = 3000, 7000

    for _ in range(10):
        points = np.random.randint(0, 10000, (num_points, 1))
        rtree = RangeTree1D(points)

        res_m = list(sorted(brute(points, x1, x2)))
        res_n = list(rtree[x1:x2])
        if list(sorted(res_n)) != list(sorted(res_m)):
            raise ValueError(f"\n{res_n}\n {res_m}\n {points}")

    print(RangeTree1D.split_value.cache_info())
    # points = np.array([[6, 1, 0, 6, 1]]).T
    # rtree = RangeTree1D(points)
    # print(rtree)
    # res_n = list(rtree[x1:x2])
    # print(res_n)
    # print(list(brute(points, x1, x2)))

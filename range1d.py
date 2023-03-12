from functools import cache
from sys import maxsize
from typing import Union, Iterator

import numpy as np
from more_itertools import only

from rangetree import RangeTree, Leaf, OUT_OF_BOUNDS, Interval


class RangeTree1D(RangeTree):
    __slots__ = ("split_value", "left", "right")

    def __init__(
        self,
        split_value: float,
        left: RangeTree,
        right: RangeTree,
    ):
        self.split_value = split_value
        self.left = left
        self.right = right

    @cache
    def find_split_value(self) -> float:
        return max(
            self.split_value,
            self.left.find_split_value(),
            self.right.find_split_value(),
        )

    def is_leaf(self):
        return self.left is None and self.right is None

    def __repr__(self):
        return f"{type(self).__name__}({self.split_value})"

    def report_leaves(self):
        yield from self.left.report_leaves()
        yield from self.right.report_leaves()

    def yield_line(self, indent: str, prefix: str) -> Iterator[str]:
        yield f"{indent}{prefix}----{self}\n"
        indent += "     " if prefix == "R" else "|    "
        yield from self.left.yield_line(indent, "L")
        yield from self.right.yield_line(indent, "R")

    @staticmethod
    def construct(values, axis=0) -> "RangeTree1D":
        return RangeTree1D.build_range_tree(values, axis)

    @staticmethod
    def build_range_tree(values: np.ndarray, axis: int) -> "RangeTree1D":
        """Build a 1D Range Tree from the bottom up and returns the root, and the nodes on the same level.
        This is just for indexing.
        It is possible to augment the structure to store any information.
        """
        # O(n log n) because of sorting

        leaves = [
            Leaf(value, axis) for value in sorted(values, key=lambda row: row[axis])
        ]
        # n + n/2 + n/4 + n/8 + ... + 1 ≅ 2n  (Geometric summation) = O(n)
        while (n_leaves := len(leaves)) > 1:
            nodes = [
                RangeTree1D(leaves[i - 1].find_split_value(), leaves[i - 1], leaves[i])
                for i in range(1, n_leaves, 2)
            ]
            if n_leaves & 1:
                nodes.append(
                    RangeTree1D(
                        leaves[n_leaves - 1].find_split_value(),
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

    def query_axis(self, interval: Interval) -> Iterator[np.array]:
        """Queries a 1D Range Tree.
        Let P be a set of n points in 1-D space. The set P
        can be stored in a balanced binary search tree, which uses O(n) storage and
        has O(n log n) construction time, such that the points in a query range can be
        reported in time O(k + log n), where k is the number of reported points.
        """

        v_split = self.find_split_node(self, interval)
        if isinstance(v_split, Leaf):
            # check if the point in v_split
            if v_split.split_value in interval:  # inclusive version
                yield from v_split.report_leaves()
        else:
            v = v_split.left
            while not isinstance(v, Leaf):
                if v.split_value >= interval.start:
                    # report right subtree
                    yield from v.right.report_leaves()
                    v = v.left
                else:
                    v = v.right
            # v is now a leaf
            if v.split_value in interval:
                yield v.point
            # now we follow right side
            v = v_split.right
            while not isinstance(v, Leaf):
                if v.split_value < interval.end:
                    # report left subtree
                    yield from v.left.report_leaves()
                    # it is possible to traverse to an external node
                    if v.right is OUT_OF_BOUNDS:
                        return
                    v = v.right
                else:
                    v = v.left
            # check whether this point should be included too
            if v.split_value in interval:
                yield from v.report_leaves()

    query = query_axis

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

        return self.query(Interval(start, stop))


def brute(points, x1, x2):
    for point in points:
        if x1 <= point < x2:
            yield point


if __name__ == "__main__":
    from random import randint

    num_points = 10000
    x1, x2 = 5000, 7000

    for _ in range(5):
        points = np.random.randint(0, 10000, (num_points, 1))
        rtree = RangeTree1D.construct(points)

        res_m = list(sorted(brute(points, x1, x2)))
        res_n = list(rtree[x1:x2])
        if list(sorted(res_n)) != list(sorted(res_m)):
            raise ValueError(f"\n{res_n}\n {res_m}\n {points}")

        # print(RangeTree1D.find_split_value.cache_info())
    # points = np.array([[6, 1, 0, 6, 1]]).T
    # rtree = RangeTree1D(points)
    # res_n = list(rtree[x1:x2])
    # print(res_n)
    # print(list(brute(points, x1, x2)))

from functools import cache
from sys import maxsize
from typing import Iterator

import numpy as np
from more_itertools import only

from rangetree import OUT_OF_BOUNDS, Leaf, RangeTree
from utils import Interval, Orthotope


class RangeTree1D(RangeTree):
    __slots__ = ("split_value", "less", "greater")

    def __init__(
        self,
        split_value: float,
        less: RangeTree,
        greater: RangeTree,
    ):
        self.split_value = split_value
        self.less = less
        self.greater = greater

    @cache
    def find_split_value(self) -> float:
        return max(
            self.split_value,
            self.less.find_split_value(),
            self.greater.find_split_value(),
        )

    def __repr__(self):
        return f"{type(self).__name__}({self.split_value})"

    def report_leaves(self):
        yield from self.less.report_leaves()
        yield from self.greater.report_leaves()

    def yield_line(self, indent: str, prefix: str) -> Iterator[str]:
        yield f"{indent}{prefix}----{self}\n"
        indent += "     " if prefix == "R" else "|    "
        yield from self.less.yield_line(indent, "L")
        yield from self.greater.yield_line(indent, "R")

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

    def query_axis_recursive(self, query_interval: Interval):
        stack = [(self, Interval(-maxsize, maxsize))]
        while stack:
            current_node, x_range = stack.pop()

            if isinstance(current_node, Leaf):
                yield from current_node.query_interval(query_interval)
            elif query_interval.contains(x_range):
                yield from current_node.report_leaves()
            elif not query_interval.is_disjoint_from(x_range):
                stack.extend(
                    (
                        (
                            current_node.greater,
                            Interval(current_node.split_value, x_range.end),
                        ),
                        (
                            current_node.less,
                            Interval(x_range.start, current_node.split_value),
                        ),
                    )
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
            v = v_split.less
            while not isinstance(v, Leaf):
                if v.split_value >= interval.start:
                    # report right subtree
                    yield from v.greater.report_leaves()
                    v = v.less
                else:
                    v = v.greater
            # v is now a leaf
            if v.split_value in interval:
                yield v.point
            # now we follow right side
            v = v_split.greater
            while not isinstance(v, Leaf):
                if v.split_value < interval.end:
                    # report left subtree
                    yield from v.less.report_leaves()
                    # it is possible to traverse to an external node
                    if v.greater is OUT_OF_BOUNDS:
                        return
                    v = v.greater
                else:
                    v = v.less
            # check whether this point should be included too
            if v.split_value in interval:
                yield from v.report_leaves()

    def query(self, item: Orthotope, axis: int = 0):
        yield from self.query_axis(item.intervals[axis])


if __name__ == "__main__":
    import doctest

    doctest.testmod()

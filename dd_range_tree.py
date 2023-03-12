from operator import itemgetter
from typing import NamedTuple

import numpy as np

from layered_range_tree import LayeredRangeTree
from range2d import RangeTree2D
from rangetree import OUT_OF_BOUNDS, Leaf, RangeTree, Interval


class HyperRectangle(NamedTuple):
    intervals: list[Interval]

    def __contains__(self, item):
        if len(item) != len(self.intervals):
            raise ValueError()
        return all(value in interval for value, interval in zip(item, self.intervals))

    def __iter__(self):
        yield from self.intervals

    @property
    def x_range(self):
        return self.intervals[0]

    @property
    def y_range(self):
        return self.intervals[1]


class DDRangeTree(RangeTree2D):
    """Uses a Layered Range tree for the base case."""

    def __init__(
        self,
        split_value: float,
        left: RangeTree,
        right: RangeTree,
        assoc: RangeTree,
        depth,
    ):
        super().__init__(split_value, left, right, assoc)
        self.depth = depth

    @staticmethod
    def construct(values: np.ndarray, axis: int = 0):
        _, max_depth = values.shape

        if max_depth == 2:
            return LayeredRangeTree.construct(values)

        sorted_by_x = np.array(sorted(values, key=itemgetter(axis)))

        def get_correct_tree(depth) -> RangeTree:
            if max_depth - depth == 3:
                return LayeredRangeTree
            return DDRangeTree

        def construct_impl(points, depth: int):
            if points.size == 0:
                return OUT_OF_BOUNDS
            elif len(points) == 1:
                return Leaf(points[0], depth)

            tree = get_correct_tree(depth).construct(points, depth + 1)
            mid = (len(points)) >> 1
            return DDRangeTree(
                points[mid][depth],
                construct_impl(points[mid:], depth),
                construct_impl(points[:mid], depth),
                tree,
                depth,
            )

        return construct_impl(sorted_by_x, axis)

    def get_correct_tree(self, depth):
        if self.depth - depth == 3:
            return LayeredRangeTree
        return DDRangeTree

    def query(self, hyper_rectangle: HyperRectangle, depth: int = 0):
        v_split = self.find_split_node(self, hyper_rectangle.x_range)
        x_range = hyper_rectangle.x_range

        if isinstance(v_split, Leaf):
            # check if the point in v_split
            if v_split.point in hyper_rectangle:
                yield v_split.point
        else:
            v = v_split.left
            while not isinstance(v, Leaf):
                if v.split_value >= x_range.start:
                    # report right subtree
                    yield from v.right.query(
                        HyperRectangle(hyper_rectangle.intervals[depth + 1 :]),
                        depth + 1,
                    )
                    v = v.left
                else:
                    v = v.right
            # v is now a leaf
            if v in hyper_rectangle:
                yield v.point
            # now we follow right side
            v = v_split.right
            while not isinstance(v, Leaf):
                if v is OUT_OF_BOUNDS:
                    return
                if v.split_value < x_range.end:
                    # report left subtree
                    yield from v.left.query(
                        HyperRectangle(hyper_rectangle.intervals[depth + 1 :]),
                        depth + 1,
                    )
                    # it is possible to traverse to an external node
                    v = v.right
                else:
                    v = v.left
            # check whether this point should be included too
            if v in hyper_rectangle:
                yield v.point


def brute_algorithm(coords, rectangle):
    for coord in coords:
        if all(x in interval for x, interval in zip(coord, rectangle)):
            yield coord


if __name__ == "__main__":
    from random import randint

    d = 5
    test_rounds = 1

    start, end = 0, 5000

    num_coords = 10
    for _ in range(100):
        points = np.random.randint(0, 10000, (3, 3))
        r2d = DDRangeTree.construct(points)
        rectangle = HyperRectangle(
            [Interval(start, end), Interval(start, end), Interval(start, end)]
        )
        result = r2d.query(rectangle)

        res_n = list(sorted([tuple(map(int, elem)) for elem in result]))
        res_m = list(sorted(brute_algorithm(points, rectangle)))

        if res_n != res_m:
            print(r2d.pretty_str())
            raise ValueError(f"\n{res_n}\n {res_m}\n {list(points)}")

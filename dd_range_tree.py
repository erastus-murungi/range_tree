from operator import itemgetter
from typing import Type

import numpy as np

from layered_range_tree import LayeredRangeTree
from range1d import RangeTree1D
from range2d import RangeTree2D
from rangetree import OUT_OF_BOUNDS, Leaf, Orthotope, RangeTree


class DDRangeTree(RangeTree2D):
    """Uses a Layered Range tree for the base case."""

    def __init__(
        self,
        split_value: float,
        left: RangeTree,
        right: RangeTree,
        assoc: RangeTree,
        depth: int,
    ):
        super().__init__(split_value, left, right, assoc)
        self.depth = depth

    @staticmethod
    def construct(values: np.ndarray, axis: int = 0):
        _, max_depth = values.shape

        if max_depth == 1:
            return RangeTree1D.construct(values)
        elif max_depth == 2:
            return LayeredRangeTree.construct(values)

        sorted_by_x = np.array(sorted(values, key=itemgetter(axis)))

        def get_correct_tree(depth) -> Type[RangeTree]:
            if max_depth - depth == 3:
                return LayeredRangeTree
            return DDRangeTree

        def construct_impl(points, depth: int):
            if points.size == 0:
                return OUT_OF_BOUNDS
            elif len(points) == 1:
                return Leaf(points[0], depth)

            tree = get_correct_tree(depth).construct(points, depth + 1)
            mid = ((len(points)) - 1) >> 1
            return DDRangeTree(
                points[mid][depth],
                construct_impl(points[: mid + 1], depth),
                construct_impl(points[mid + 1 :], depth),
                tree,
                depth,
            )

        return construct_impl(sorted_by_x, axis)

    def query(self, hyper_rectangle: Orthotope, depth: int = 0):
        v_split = self.find_split_node(self, hyper_rectangle.x_range)
        x_range = hyper_rectangle.x_range

        if isinstance(v_split, Leaf):
            # check if the point in v_split
            if v_split.point in hyper_rectangle:
                yield v_split.point
        else:
            v = v_split.less
            while not isinstance(v, Leaf):
                if v.split_value >= x_range.start:
                    # report right subtree
                    yield from v.greater.query(
                        hyper_rectangle,
                    )
                    v = v.less
                else:
                    v = v.greater
            # v is now a leaf
            if v.point in hyper_rectangle:
                yield v.point
            # now we follow right side
            v = v_split.greater
            while not isinstance(v, Leaf):
                if v is OUT_OF_BOUNDS:
                    return
                if v.split_value < x_range.end:
                    # report left subtree
                    yield from v.less.query(hyper_rectangle)
                    # it is possible to traverse to an external node
                    v = v.greater
                else:
                    v = v.less
            # check whether this point should be included too
            if v.point in hyper_rectangle:
                yield v.point


if __name__ == "__main__":
    import doctest

    doctest.testmod()

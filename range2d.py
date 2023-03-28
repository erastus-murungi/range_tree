from operator import itemgetter
from typing import Iterator

import numpy as np

from range1d import RangeTree1D
from rangetree import OUT_OF_BOUNDS, Leaf, RangeTree
from utils import Orthotope


class RangeTree2D(RangeTree1D):
    """Interesting geometric data structure."""

    __slots__ = "assoc"

    def __init__(
        self,
        split_value: float,
        left: RangeTree,
        right: RangeTree,
        assoc: RangeTree | np.ndarray,
    ):
        super().__init__(split_value, left, right)
        self.assoc = assoc

    @staticmethod
    def construct(values: np.ndarray, axis=0) -> RangeTree:
        def construct_impl(points: np.ndarray, axis) -> RangeTree:
            """Build a 2D Range Tree recursively.
            It is compressed so that Internal nodes also contain some pointers"""
            if points.size == 0:
                return OUT_OF_BOUNDS
            elif len(points) == 1:
                return Leaf(points[0], axis)

            mid = ((len(points)) - 1) >> 1
            return RangeTree2D(
                points[mid][axis],
                construct_impl(points[: mid + 1], axis),
                construct_impl(points[mid + 1 :], axis),
                RangeTree1D.construct(points, axis=axis + 1),
            )

        return construct_impl(np.array(sorted(values, key=itemgetter(axis))), axis)

    def query(self, box: Orthotope, axis=0) -> Iterator[np.ndarray]:
        """A query with an axis-parallel rectangle in a range tree storing n
        points takes O(log2 n+k) time, where k is the number of reported points
        """

        # find v_split using the x_axis
        v_split = self.find_split_node(self, box.x_range)
        if isinstance(v_split, Leaf):
            # check if v_split must be reported
            if v_split.point in box:
                # check if v_split must be reported
                yield v_split.point
        else:
            # (∗ Follow the path to x and call 1D_RANGE_QUERY on the subtrees right of the path. ∗)
            v = v_split.less
            while not isinstance(v, Leaf):
                if box.x_range.start <= v.split_value:
                    yield from v.greater.assoc.query(box, axis + 1)
                    v = v.less
                else:
                    v = v.greater

            #  Check if the point stored at ν must be reported.
            if v.point in box:
                yield v.point

            # traverse the right subtree of v_split
            v = v_split.greater
            while not isinstance(v, Leaf):
                if v is OUT_OF_BOUNDS:
                    return
                if v.split_value < box.x_range.end:
                    yield from v.less.assoc.query(box, axis + 1)
                    v = v.greater
                else:
                    v = v.less

            if v.point in box:
                yield v.point


if __name__ == "__main__":
    import doctest

    doctest.testmod()

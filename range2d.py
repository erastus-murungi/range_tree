from typing import NamedTuple, Iterator

import numpy as np

from range1d import RangeTree1D
from rangetree import RangeTree, OUT_OF_BOUNDS, Leaf, Interval

__author__ = "Erastus Murungi"
__email__ = "murungi@mit.edu"


class Rectangle(NamedTuple):
    x_range: Interval
    y_range: Interval

    def __contains__(self, item):
        x, y = item
        return x in self.x_range and y in self.y_range


class RangeTree2D(RangeTree1D):
    """Interesting geometric data structure."""

    __slots__ = "assoc"

    def __init__(
        self,
        split_value: float,
        left: RangeTree,
        right: RangeTree,
        assoc: RangeTree1D | np.ndarray,
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

            mid = (len(points)) >> 1
            return RangeTree2D(
                points[mid][axis],
                construct_impl(points[:mid], axis),
                construct_impl(points[mid:], axis),
                RangeTree1D.construct(points, axis=axis),
            )

        return construct_impl(
            np.array(sorted(values, key=lambda row: row[axis])), axis + 1
        )

    def query(self, box: Rectangle) -> Iterator[np.ndarray]:
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
            v = v_split.left
            while not isinstance(v, Leaf):
                if box.x_range.start <= v.split_value:
                    yield from v.right.assoc.query_axis(box.y_range)
                    v = v.left
                else:
                    v = v.right

            #  Check if the point stored at ν must be reported.
            if v.point in box:
                yield v.point

            # traverse the right subtree of v_split
            v = v_split.right
            while not isinstance(v, Leaf):
                if v is OUT_OF_BOUNDS:
                    return
                if v.split_value < box.x_range.end:
                    yield from v.left.assoc.query_axis(box.y_range)
                    v = v.right
                else:
                    v = v.left

            if v.point in box:
                yield v.point


if __name__ == "__main__":
    from random import randint

    def brute_algorithm(coords, x1, x2, y1, y2):
        for x, y in coords:
            if x1 <= x < x2 and y1 <= y < y2:
                yield x, y

    x1, x2, y1, y2 = -1, 3000, 0, 8000

    num_coords = 10
    for _ in range(10):
        points = np.random.randint(0, 10000, (num_coords, 2))
        r2d = RangeTree2D.construct(points)
        result = r2d.query(Rectangle(Interval(x1, x2), Interval(y1, y2)))

        res_n = list(sorted([tuple(map(int, elem)) for elem in result]))
        res_m = list(sorted(brute_algorithm(points, x1, x2, y1, y2)))

        if res_n != res_m:
            print(r2d.pretty_str())
            raise ValueError(f"\n{res_n}\n {res_m}\n {list(points)}")

    # import matplotlib.pyplot as plt
# from matplotlib.patches import Rectangle

#     points = np.array([[2054, 8435],[ 595, 9691], [ 427, 3716]])
#
#     r2d = RangeTree2D.construct(points)
#     result = r2d.query(Rectangle(Interval(x1, x2), Interval(y1, y2)))
# #
#     res_n = list(sorted([tuple(map(int, elem)) for elem in result]))
#     res_m = list(sorted(brute_algorithm(points, x1, x2, y1, y2)))

# plt.scatter(points[:, 0], points[:, 1], label='all')
# plt.scatter(np.array(res_n)[:, 0], np.array(res_n)[:, 1], label='range tree')
# plt.scatter(np.array(res_m)[:, 0], np.array(res_m)[:, 1], label='brute force')
# plt.gca().add_patch(Rectangle((x1, y1), x2 - x1, y2 - y1, facecolor="none", ec='k', lw=2))
# plt.legend()
# plt.show()

# for point in r2d.query(
#     Rectangle(Interval(x1, x2), Interval(y1, y2))
# ):
#     print(point)
#
# if res_n != res_m:
#     print(r2d.pretty_str())
#     raise ValueError(f"\n{res_n}\n {res_m}\n {list(points)}")

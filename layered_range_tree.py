from bisect import bisect_left
from itertools import takewhile
from operator import itemgetter
from typing import Iterator, Optional

import numpy as np

from range2d import RangeTree2D
from rangetree import OUT_OF_BOUNDS, Interval, Leaf, Orthotope, RangeTree

LEFT, RIGHT = 1, 2


def find_ge(array, x, key) -> Optional[int]:
    """Find leftmost item greater than or equal to x"""
    idx = bisect_left(array, x, key=key)
    if idx != len(array):
        return idx
    return -1


class LayeredRangeTree(RangeTree2D):
    """Layered Range Tree with fractional cascading."""

    def __init__(
        self,
        split_value: float,
        left: RangeTree,
        right: RangeTree,
        assoc: RangeTree | np.ndarray,
        depth,
    ):
        super().__init__(split_value, left, right, assoc)
        self.depth = depth

    @staticmethod
    def construct(values: np.ndarray, axis: int = 0):
        def construct_impl(assoc: np.ndarray, axis: int):
            """Recursively build's a 1D range tree with fractional cascading.
            This version only searches for the pointer at the beginning during the binary search step.
            Only the leaves store the full coordinates. Internal nodes only store splitting values"""
            # sort by y_values

            if assoc.size == 0:
                return OUT_OF_BOUNDS
            elif len(assoc) == 1:
                return Leaf(assoc[0, :-2], axis)
            else:
                mid = (len(assoc) - 1) // 2
                split_x_value = assoc[mid, axis]
                y_axis = axis + 1

                assoc_sorted = np.array(sorted(assoc, key=itemgetter(y_axis)))

                assoc_left, assoc_right = assoc[: mid + 1, :], assoc[mid + 1 :, :]
                assoc_left_sorted = np.array(sorted(assoc_left, key=itemgetter(y_axis)))
                assoc_right_sorted = np.array(
                    sorted(assoc_right, key=itemgetter(y_axis))
                )

                assoc_sorted[:, y_axis + LEFT] = [
                    find_ge(assoc_left_sorted, x, itemgetter(y_axis))
                    for x in assoc_sorted[:, y_axis]
                ]
                assoc_sorted[:, y_axis + RIGHT] = [
                    find_ge(assoc_right_sorted, x, itemgetter(y_axis))
                    for x in assoc_sorted[:, y_axis]
                ]

                return LayeredRangeTree(
                    split_x_value,
                    construct_impl(
                        assoc_left,
                        axis,
                    ),
                    construct_impl(
                        assoc_right,
                        axis,
                    ),
                    assoc_sorted,
                    axis,
                )

        pointers = np.full((len(values), 2), -1)
        sorted_by_x = np.array(sorted(values, key=itemgetter(axis)))
        assoc = np.c_[sorted_by_x, pointers]

        return construct_impl(assoc, axis)

    def report_nodes(
        self, v: RangeTree, box: Orthotope, frm: int
    ) -> Iterator[np.array]:
        # report left subtree
        if isinstance(v, Leaf):
            yield from v.query(box)
        else:
            yield from takewhile(
                lambda element: element[self.depth + 1] < box.y_range.end,
                v.assoc[frm:, :-2],
            )

    def query(self, box: Orthotope, axis=0):
        v_split = RangeTree.find_split_node(self, box.x_range)
        if isinstance(v_split, Leaf):
            # check if the point in v_split, leaves have no pointers
            if v_split.point in box:
                yield v_split.point
        else:
            y_axis = self.depth + 1
            idx = find_ge(
                v_split.assoc,
                box.y_range.start,
                key=itemgetter(y_axis),
            )
            if idx == (-1):
                return
            # (∗ Follow the path to x and call 1D_RANGE_QUERY on the subtrees right of the path. ∗)
            v = v_split.left
            idx_ge = v_split.assoc[idx, y_axis + LEFT]
            while idx_ge != (-1) and not isinstance(v, Leaf):
                if box.x_range.start <= v.split_value:
                    # report right subtree
                    yield from self.report_nodes(
                        v.right, box, v.assoc[idx_ge, y_axis + RIGHT]
                    )
                    idx_ge = v.assoc[idx_ge, y_axis + LEFT]
                    v = v.left
                else:
                    idx_ge = v.assoc[idx_ge, y_axis + RIGHT]
                    v = v.right

            if isinstance(v, Leaf) and v.point in box:
                yield v.point

            # now we follow right side
            v = v_split.right
            idx_ge = v_split.assoc[idx, y_axis + RIGHT]
            while idx_ge != (-1) and not isinstance(v, Leaf):
                if v.split_value < box.x_range.end:
                    # report left subtree
                    yield from self.report_nodes(
                        v.left, box, v.assoc[idx_ge, y_axis + LEFT]
                    )
                    idx_ge = v.assoc[idx_ge, y_axis + RIGHT]
                    v = v.right
                else:
                    idx_ge = v.assoc[idx_ge, y_axis + LEFT]
                    v = v.left

            # check whether this point should be included too
            if isinstance(v, Leaf) and v.point in box:
                yield v.point


if __name__ == "__main__":

    def brute_algorithm(coords, x1, x2, y1, y2):
        for x, y in coords:
            if x1 <= x < x2 and y1 <= y < y2:
                yield x, y

    x1, x2, y1, y2 = -1, 2000, 0, 8000

    # points = np.array([(4728, 2874), (4728, 473), (2598, 4011)])

    # points = np.array([(1063, 6064), (1384, 477), (1298, 3993)])
    points = np.array(
        [
            (3487, 607),
            (2991, 8962),
            (6407, 3061),
            (7487, 7634),
            (52, 583),
            (9966, 8363),
            (2349, 7102),
            (1837, 4909),
            (7362, 8135),
            (3978, 5650),
            (1753, 5981),
            (1021, 2656),
            (5343, 8974),
            (3747, 1250),
        ]
    )

    # r2d = LayeredRangeTree.construct(points)
    # result = r2d.query(Rectangle(Interval(x1, x2), Interval(y1, y2)))
    # res_n = list(sorted([tuple(map(int, elem)) for elem in result]))
    # res_m = list(sorted(brute_algorithm(points, x1, x2, y1, y2)))
    # if res_n != res_m:
    #     print(r2d.pretty_str())
    #     raise ValueError(
    #         f"\n{res_n}\n {res_m}\n {[tuple(map(int, elem)) for elem in points]}"
    #     )

    num_coords = 3
    for _ in range(100):
        points = np.random.randint(0, 10000, (num_coords, 2))
        r2d = LayeredRangeTree.construct(points)
        result = list(r2d.query(Orthotope([Interval(x1, x2), Interval(y1, y2)])))

        res_n = list(sorted([tuple(map(int, elem)) for elem in result]))
        res_m = list(sorted(brute_algorithm(points, x1, x2, y1, y2)))

        if res_n != res_m:
            print(r2d.pretty_str())
            raise ValueError(
                f"\n{res_n}\n {res_m}\n {[tuple(map(int, elem)) for elem in points]}"
            )

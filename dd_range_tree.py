from operator import itemgetter
from typing import Type

import numpy as np

from layered_range_tree import LayeredRangeTree
from range1d import RangeTree1D
from range2d import RangeTree2D
from rangetree import OUT_OF_BOUNDS, Leaf, RangeTree, Interval, HyperRectangle


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
                        hyper_rectangle,
                    )
                    v = v.left
                else:
                    v = v.right
            # v is now a leaf
            if v.point in hyper_rectangle:
                yield v.point
            # now we follow right side
            v = v_split.right
            while not isinstance(v, Leaf):
                if v is OUT_OF_BOUNDS:
                    return
                if v.split_value < x_range.end:
                    # report left subtree
                    yield from v.left.query(hyper_rectangle)
                    # it is possible to traverse to an external node
                    v = v.right
                else:
                    v = v.left
            # check whether this point should be included too
            if v.point in hyper_rectangle:
                yield v.point


def brute_algorithm(coords, rectangle):
    for coord in coords:
        if all(x in interval for x, interval in zip(coord, rectangle)):
            yield tuple(coord)


if __name__ == "__main__":

    start, end = 0, 7000

    dimensions = 4
    n_coords = 20
    for _ in range(100):
        points = np.random.randint(0, 10000, (n_coords, dimensions))
        r2d = DDRangeTree.construct(points)
        rectangle = HyperRectangle([Interval(start, end) for _ in range(dimensions)])
        result = r2d.query(rectangle)

        res_n = list(sorted([tuple(map(int, elem)) for elem in result]))
        res_m = list(sorted(brute_algorithm(points, rectangle)))

        if res_n != res_m:
            print(r2d.pretty_str())
            raise ValueError(
                f"\n{res_n}\n {res_m}\n {[tuple(map(int, elem)) for elem in points]}"
            )

    # points = np.array(
    #     [
    #         (658, 887, 4316, 3197),
    #         (5253, 8682, 8034, 8963),
    #         (545, 8267, 8317, 1488),
    #         (2237, 709, 8149, 9034),
    #         (6351, 7637, 2006, 8695),
    #         (2881, 3866, 5785, 8742),
    #         (3930, 9347, 8664, 1383),
    #         (7057, 9274, 7207, 6310),
    #         (3562, 6999, 5495, 1149),
    #         (4381, 8845, 2782, 1130),
    #     ]
    # )
    # r2d = DDRangeTree.construct(points)
    # print(r2d.pretty_str())
    # rectangle = HyperRectangle([Interval(start, end) for _ in range(points.shape[1])])
    # result = r2d.query(rectangle)
    # # for r in result:
    # #     print(r)
    #
    # res_n = list(sorted([tuple(map(int, elem)) for elem in result]))
    # res_m = list(sorted(brute_algorithm(points, rectangle)))
    #
    # if res_n != res_m:
    #     raise ValueError(
    #         f"\n{res_n}\n {res_m}\n {[tuple(map(int, elem)) for elem in points]}"
    #     )

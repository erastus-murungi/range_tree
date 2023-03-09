import numpy as np

from range1d import RangeTree1D
from rangetree import RangeTree, Split, OUT_OF_BOUNDS, Leaf

__author__ = "Erastus Murungi"
__email__ = "murungi@mit.edu"


class Split2D(Split):
    __slots__ = "assoc"

    def __init__(self, split_value, left, right, assoc):
        super().__init__(split_value, left, right)
        self.assoc = assoc


Node2D = Split2D | Leaf


class RangeTree2D(RangeTree):
    """Interesting geometric data structure."""

    def __init__(self, values: np.ndarray):
        super().__init__(
            self.build_range_tree2d(np.array(sorted(values, key=lambda row: row[0])))
        )

    def build_range_tree2d(self, points: np.ndarray) -> Node2D:
        """Build a 2D Range Tree recursively.
        It is compressed so that Internal nodes also contain some pointers"""
        if points.size == 0:
            return OUT_OF_BOUNDS
        elif len(points) == 1:
            return Leaf(points[0])

        mid = (len(points)) >> 1
        return Split2D(
            points[mid][0],
            self.build_range_tree2d(points[:mid]),
            self.build_range_tree2d(points[mid:]),
            RangeTree1D(points, axis=1),
        )

    def query_2d_range_tree(self, x1, x2, y1, y2):
        """A query with an axis-parallel rectangle in a range tree storing n
        points takes O(log2 n+k) time, where k is the number of reported points
        """

        # partial functions to deal with edge cases
        get_x = lambda node: node.point[0]
        get_y = lambda node: node.point[1]

        # find v_split using the x_axis
        v_split = self.find_split_node(x1, x2)
        if isinstance(v_split, Leaf):
            # check if v_split must be reported
            if x1 <= get_x(v_split) < x2 and y1 <= get_y(v_split) < y2:
                # check if v_split must be reported
                yield v_split.point
        else:
            # (∗ Follow the path to x and call 1D_RANGE_QUERY on the subtrees right of the path. ∗)
            v = v_split.left
            while not isinstance(v, Leaf):
                if x1 <= v.split_value:
                    if v.right is not OUT_OF_BOUNDS:
                        if isinstance(v.right, Leaf):
                            if y1 <= get_y(v.right) < y2:
                                yield v.right.point
                        else:
                            yield from v.right.assoc.query_range_tree1d(y1, y2)
                    v = v.left
                else:
                    v = v.right

            #  Check if the point stored at ν must be reported.
            if x1 <= get_x(v) < x2 and y1 <= get_y(v) < y2:
                yield v.point

            # traverse the right subtree of v_split
            v = v_split.right
            while v is not OUT_OF_BOUNDS and not isinstance(v, Leaf):
                if v.split_value < x2:
                    if isinstance(v.left, Leaf):
                        if y1 <= v.left.point[1] < y2:
                            yield v.left.point
                    else:
                        yield from v.left.assoc.query_range_tree1d(y1, y2)
                    v = v.right
                else:
                    v = v.left

            if v is not OUT_OF_BOUNDS and x1 <= get_x(v) < x2 and y1 <= get_y(v) < y2:
                yield v.point


if __name__ == "__main__":
    from random import randint


    def brute_algorithm(coords, x1, x2, y1, y2):
        for x, y in coords:
            if x1 <= x < x2 and y1 <= y < y2:
                yield x, y


    x1, x2, y1, y2 = -1, 3000, 0, 8000

    num_coords = 10000
    for _ in range(5):
        points = np.random.randint(0, 10000, (num_coords, 2))
        r2d = RangeTree2D(points)
        result = r2d.query_2d_range_tree(x1, x2, y1, y2)

        res_n = list(sorted([tuple(map(int, elem)) for elem in result]))
        res_m = list(sorted(brute_algorithm(points, x1, x2, y1, y2)))

        if res_n != res_m:
            print(r2d.pretty_print_tree())
            raise ValueError(f"\n{res_n}\n {res_m}\n {list(points)}")

    # import matplotlib.pyplot as plt
# from matplotlib.patches import Rectangle
# points = np.array(
#     [(9339, 6524), (2549, 5561), (3881, 3035), (9763, 2979), (4128, 786), (6763, 9856), (5504, 3081), (7080, 603),
#      (9686, 2104), (3512, 9812)]
# )

# r2d = RangeTree2D(points)
# result = r2d.query_2d_range_tree(x1, x2, y1, y2)
#
# res_n = list(sorted([tuple(map(int, elem)) for elem in result]))
# res_m = list(sorted(brute_algorithm(points, x1, x2, y1, y2)))

# plt.scatter(points[:, 0], points[:, 1], label='all')
# plt.scatter(np.array(res_n)[:, 0], np.array(res_n)[:, 1], label='range tree')
# plt.scatter(np.array(res_m)[:, 0], np.array(res_m)[:, 1], label='brute force')
# plt.gca().add_patch(Rectangle((x1, y1), x2 - x1, y2 - y1, facecolor="none", ec='k', lw=2))
# plt.legend()
# plt.show()

# for point in r2d.query_2d_range_tree(
#     x1, x2, y1, y2
# ):
#     print(point)

# if res_n != res_m:
#     print(r2d.pretty_print_tree())
#     raise ValueError(f"\n{res_n}\n {res_m}\n {list(points)}")

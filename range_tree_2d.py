from range1d import RangeTree1D, Node, Leaf
from typing import Union
from dataclasses import dataclass
from rangetree import RangeTree
from functools import partial

__author__ = 'Erastus Murungi'
__email__ = 'murungi@mit.edu'


@dataclass
class Node2D(Node):
    y_tree: RangeTree1D


@dataclass
class Leaf2D(Leaf):
    y_tree: RangeTree1D


class RangeTree2D(RangeTree):
    """Interesting geometric data structure."""
    def __init__(self, values):
        super().__init__()
        self.root = self.build_range_tree2d(sorted(values, key=lambda w: w[0]))

    @staticmethod
    def isleaf(node):
        return type(node) == Leaf2D or type(node) == Leaf

    def build_range_tree2d(self, points, parent=None) -> Union[Leaf2D, Node2D, None]:
        """ Build a 2D Range Tree recursively """
        if points:
            y_t = RangeTree1D(points, axis=1)
            # base case
            if len(points) == 1:
                v = Leaf2D(points[0], parent, y_t)  # double information storage
                return v
            else:
                mid = (len(points)) >> 1
                v = Node2D(None, None, points[mid], parent, y_t)
                v_left = self.build_range_tree2d(points[:mid], parent=v)
                v_right = self.build_range_tree2d(points[mid + 1:], parent=v)
                v.left = v_left
                v.right = v_right
                return v
        return None

    def query_2d_range_tree(self, x1, x2, y1, y2):
        """ A query with an axis-parallel rectangle in a range tree storing n
         points takes O(log2 n+k) time, where k is the number of reported points
        """

        if x1 > x2:
            x1, x2 = x2, x1  # swap
        if y1 > y2:
            y1, y2 = y2, y1

        output = []

        # partial functions to deal with edge cases
        get_y1d = partial(self.__get_point_1d, 1)  # y coordinate for 1D tree
        get_x2d = partial(self.__get_point_2d, 0)  # x coordinate for 2D tree
        get_y2d = partial(self.__get_point_2d, 1)  # y coordinate for 2D tree

        # find v_split using the x_axis
        v_split = self.find_split_node(x1, x2, getpoint=get_x2d)
        if x1 <= get_x2d(v_split) < x2 and y1 <= get_y2d(v_split) < y2:
            # check if v_split must be reported
            output.append(v_split.point)
        if self.isleaf(v_split):
            return output

        # (∗ Follow the path to x and call 1D_RANGE_QUERY on the subtrees right of the path. ∗)
        v = v_split.left
        while not self.isleaf(v):
            if get_x2d(v) >= x1:
                # predicate necessary because in 2D range trees, even the internal nodes store data
                if y1 <= get_y2d(v) < y2:
                    output.append(v.point)
                if v.right is not None:
                    subset = v.right.y_tree.query_range_tree1d(y1, y2, get_y1d)
                    output += subset
                v = v.left
            else:
                v = v.right

        #  Check if the point stored at ν must be reported.
        if x1 <= get_x2d(v) < x2 and y1 <= get_y2d(v) < y2:
            output.append(v.point)

        # traverse the right subtree of v_split
        v = v_split.right
        while v is not None and not self.isleaf(v):
            if get_x2d(v) < x2:
                if y1 <= get_y2d(v) < y2:
                    output.append(v.point)
                subset = v.left.y_tree.query_range_tree1d(y1, y2, get_y1d)
                output += subset
                v = v.right
            else:
                v = v.left

        if v is not None and x1 <= get_x2d(v) < x2 and y1 <= get_y2d(v) < y2:
            output.append(v.point)

        return output

    @staticmethod
    def __get_point_1d(axis, w):
        if type(w) == Leaf:
            return w.point[axis]
        else:
            return w.point

    @staticmethod
    def __get_point_2d(axis, w):
        return w.point[axis]


def brute_algorithm(coords, x1, x2, y1, y2):
    for x, y in coords:
        if x1 <= x < x2 and y1 <= y < y2:
            yield x, y


if __name__ == '__main__':
    from random import randint
    lim = 100_000

    def randy():
        yield randint(0, lim)

    test_rounds = 300
    num_coords = 100
    x1, x2, y1, y2 = 0, 10_000, 0, 50_000
    for _ in range(test_rounds):
        coordinates = [tuple([next(randy()), next(randy())]) for _ in range(num_coords)]
        r2d = RangeTree2D(coordinates)
        result = r2d.query_2d_range_tree(x1, x2, y1, y2)
        gen = r2d.report(result)

        range_list = list(gen)
        brute = list(brute_algorithm(coordinates, x1, x2, y1, y2))
        # print(range_list, '\n', brute)
        print(len(brute), len(range_list))
        if len(range_list) != len(brute):
            raise ValueError()

from range1d import Node, Leaf
from dataclasses import dataclass
from typing import Iterable
from rangetree import *


@dataclass
class Node2D(Node):
    sorted_arr: Iterable


@dataclass
class Leaf2D(Leaf):
    sorted_arr: Iterable


class LayeredRangeTree(RangeTree):
    """Layered Range Tree with fractional cascading."""

    def __init__(self, values):
        super().__init__()
        self.root = self.build_layered_range_tree(values)

    @staticmethod
    def isleaf(node):
        return type(node) == Leaf or type(node) == Leaf2D

    def build_layered_range_tree(self, values):
        points = sorted(values, key=by_x)
        assoc_root = list(map(lambda w: [w, [None, None]], sorted(points, key=by_y)))
        return self.build_layered_range_tree_helper(points, assoc_root)

    def build_layered_range_tree_helper(self, points, assoc, parent=None):
        # sort by y_values
        if points:
            if len(points) == 1:
                v = Leaf2D(points[0], parent, assoc)
                return v
            else:
                mid = (len(points)) >> 1

                # prep the cascaded points
                assoc_left = list(map(lambda w: [w, [-1, -1]], sorted(points[:mid], key=by_y)))
                assoc_right = list(map(lambda w: [w, [-1, -1]], sorted(points[mid:], key=by_y)))

                self.fractional_cascade(assoc, assoc_left, get_y, LEFT)
                self.fractional_cascade(assoc, assoc_right, get_y, RIGHT)

                # close modifications
                assoc = list(map(lambda y: (y[0], tuple(y[1])), assoc))
                v = Node2D(None, None, points[mid - 1][0], parent, assoc)  # store the max in the left subtree

                v.left = self.build_layered_range_tree_helper(points[:mid], assoc_left, parent=v)
                v.right = self.build_layered_range_tree_helper(points[mid:], assoc_right, parent=v)

                return v
        return None

    @staticmethod
    def fractional_cascade(p_assoc, assoc_child, getpoint, direction):
        """Set the pointers in the main array
        correctness to be proven"""

        x = 0
        nl, n = len(assoc_child), len(p_assoc)
        i, j = n - 1, nl - 1
        while j >= 0 and i >= 0:
            if getpoint(p_assoc[i]) <= getpoint(assoc_child[j]):
                p_assoc[i][1][direction] = j
                if getpoint(assoc_child[j - 1]) == getpoint(p_assoc[i - 1]):
                    j -= 1
                i -= 1
            else:
                i -= 1
                x += 1

    @staticmethod
    def bin_search_low(A, getpoint, x: float):
        """Searches for the smallest key >= x"""
        if getpoint(A[-1]) < x:
            return None
        high = len(A) - 1
        low = 0
        while low <= high:
            mid = (high + low) >> 1
            if getpoint(A[mid]) == x:
                return mid
            elif x > getpoint(A[mid]):
                low = mid + 1
            else:
                high = mid - 1
        return low

    def query_layered_range_tree(self, x1, x2, y1, y2):
        pass


if __name__ == '__main__':
    # import numpy as np
    #
    # points = np.random.randint(100, size=50)
    # points.sort()
    # key = 56
    # print(points)
    # k = bin_search_low(points, key)
    # if k is not None:
    #     print(k, points[k])
    from random import randint

    lim = 100


    def randy():
        yield randint(0, lim)


    test_rounds = 1
    num_coords = 10
    # x1, x2, y1, y2 = 0, 10_000, 0, 50_000
    for _ in range(test_rounds):
        # coordinates = [tuple([next(randy()), next(randy())]) for _ in range(num_coords)]
        coordinates = [(2, 19), (5, 80), (7, 10), (8, 37), (12, 3), (15, 99), (17, 62),
                       (21, 49), (33, 30), (41, 95), (52, 23), (58, 59), (67, 89), (93, 70)]
        r2d = LayeredRangeTree(coordinates)
        print(r2d)
        # result = r2d.query_2d_range_tree(x1, x2, y1, y2)
        # gen = r2d.report(result)
        #
        # range_list = list(gen)
        # brute = list(brute_algorithm(coordinates, x1, x2, y1, y2))
        # # print(range_list, '\n', brute)
        # print(len(brute), len(range_list))
        # if len(range_list) != len(brute):
        #     raise ValueError()

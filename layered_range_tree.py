from range1d import Node, Leaf
from dataclasses import dataclass
from rangetree import *


@dataclass
class Node2D(Node):
    assoc: list


@dataclass
class Leaf2D(Leaf):
    assoc: list


class LayeredRangeTree(RangeTree):
    """Layered Range Tree with fractional cascading."""

    def __init__(self, values):
        super().__init__()
        self.root: Node2D = self.build_layered_range_tree(values)

    @staticmethod
    def isleaf(node):
        return type(node) == Leaf or type(node) == Leaf2D

    def build_layered_range_tree(self, values):
        points = sorted(values, key=by_x)
        assoc_root = list(map(lambda w: [w, [-1, -1]], sorted(points, key=by_y)))
        return self.build_layered_range_tree_helper(points, assoc_root)

    def build_layered_range_tree_helper(self, points, assoc, parent=None):
        """Recursively build's a 1D range tree with fractional cascading.
        This version only searches for the pointer at the beginning during the binary search step.
        Only the leaves store the full coordinates. Internal nodes only store splitting values"""
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

                # store the max in the left subtree, the max is at loc mid - 1
                v = Node2D(None, None, points[mid - 1][0], parent, assoc)

                v.left = self.build_layered_range_tree_helper(points[:mid], assoc_left, parent=v)
                v.right = self.build_layered_range_tree_helper(points[mid:], assoc_right, parent=v)

                return v
        return None

    @staticmethod
    def fractional_cascade(p_assoc, assoc_child, getpoint, direction):
        """Set the pointers in the array stored at the parent. p_assoc is the y-sorted array which is
        split to two assoc_child arrays"""

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

    @staticmethod
    def bin_search_low(A, getpoint, x: float):
        """Searches for the smallest key >= x"""
        if getpoint(A[-1]) < x:
            return len(A) - 1
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

    @staticmethod
    def filter_by_y(node, i, y2):
        """Find the values which are less than y"""
        assert i is not None
        if i >= len(node.assoc) or i is None:
            return []
        else:
            h = i
            n = len(node.assoc) - i
            sub = []
            while h < n and get_y(node.assoc[h]) < y2:
                sub.append(node.assoc[h][0])
                h += 1
            return sub

    def query_layered_range_tree(self, x1, x2, y1, y2):
        if y1 > y2:
            y1, y2 = y2, y1

        output = []

        v_split = self.find_split_node(x1, x2)
        v_i = self.bin_search_low(v_split.assoc, get_y, y1)

        i = v_i
        if self.isleaf(v_split):
            # check if the point in v_split, leaves have no pointers
            if x1 <= by_x(v_split.point) < x2 and y1 <= by_y(v_split.point) < y2:
                output.append(v_split.point)
        else:
            i = trickle_down(v_split, i, LEFT)
            v = v_split.left

            while not self.isleaf(v):
                if v.point >= x1:
                    # report right subtree
                    a = trickle_down(v, i, RIGHT)
                    subset = self.filter_by_y(v.right, a, y2)
                    output += subset

                    i = trickle_down(v, i, LEFT)
                    v = v.left
                else:
                    i = trickle_down(v, i, RIGHT)
                    v = v.right

            # v is now a leaf
            if y1 <= by_y(v.point) < y2 and x1 <= by_x(v.point) < x2:
                output.append(v.point)
            # now we follow right side
            i = trickle_down(v_split, v_i, RIGHT)
            v = v_split.right
            while v is not None and not self.isleaf(v):
                if v.point < x2:
                    # report left subtree
                    a = trickle_down(v, i, LEFT)
                    subset = self.filter_by_y(v.left, a, y2)
                    output += subset

                    i = trickle_down(v, i, RIGHT)
                    v = v.right

                else:
                    i = trickle_down(v, i, LEFT)
                    v = v.left

            # check whether this point should be included too
            if v is not None and y1 <= by_y(v.point) < y2 and x1 <= by_x(v.point) < x2:
                output.append(v.point)
        return output


def brute_algorithm(coords, x1, x2, y1, y2):
    for x, y in coords:
        if x1 <= x < x2 and y1 <= y < y2:
            yield x, y


if __name__ == '__main__':
    from random import randint

    lim = 200


    def randy():
        yield randint(0, lim)


    test_rounds = 100_000
    num_coords = 8
    x1, x2, y1, y2 = 0, 150, 0, 100
    for _ in range(test_rounds):
        # coordinates = [tuple([next(randy()), next(randy())]) for _ in range(num_coords)]
        # coordinates = [(2, 19), (5, 80), (7, 10), (8, 37), (12, 3), (15, 99), (17, 62),
        #                (21, 49), (33, 30), (41, 95), (52, 23), (58, 59), (67, 89), (93, 70)]
        coordinates = [(95, 0), (117, 63), (35, 175), (82, 186), (84, 60), (114, 63), (200, 63), (157, 190)]

        r2d = LayeredRangeTree(coordinates)
        print(r2d)
        rep = r2d.query_layered_range_tree(x1, x2, y1, y2)
        range_list = list(rep)
        brute = list(brute_algorithm(coordinates, x1, x2, y1, y2))
        # print(range_list, '\n', brute)
        if len(range_list) != len(brute):
            print(len(brute), len(range_list))
            print(sorted(coordinates))
            raise ValueError()

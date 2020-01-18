from range1d import Node, Leaf
from dataclasses import dataclass
from rangetree import *
from operator import itemgetter

LEFT, RIGHT = 0, 1
COORD_INDEX, POINTER_INDEX = 0, 1


@dataclass
class Node2D(Node):
    assoc: list


@dataclass
class Leaf2D(Leaf):
    assoc: list


class LayeredRangeTree(RangeTree):
    """Layered Range Tree with fractional cascading."""

    def __init__(self, values, depth=0):
        super().__init__()
        self.depth = depth
        self.by_x = itemgetter(depth)
        self.by_y = itemgetter(depth + 1)
        self.root: Node2D = self.build_layered_range_tree(values)

    @staticmethod
    def isleaf(node):
        return type(node) == Leaf or type(node) == Leaf2D

    def build_layered_range_tree(self, values):
        points = sorted(values, key=self.by_x)
        assoc_root = list(map(lambda w: [w, [-1, -1]], sorted(points, key=self.by_y)))
        return self.build_layered_range_tree_helper(points, assoc_root)

    def build_layered_range_tree_helper(self, points, assoc):
        """Recursively build's a 1D range tree with fractional cascading.
        This version only searches for the pointer at the beginning during the binary search step.
        Only the leaves store the full coordinates. Internal nodes only store splitting values"""
        # sort by y_values

        if points:
            if len(points) == 1:
                v = Leaf2D(points[0], assoc)
                return v
            else:
                mid = (len(points)) >> 1

                # prep the cascaded points
                assoc_left = list(map(lambda w: [w, [-1, -1]], sorted(points[:mid], key=self.by_y)))
                assoc_right = list(map(lambda w: [w, [-1, -1]], sorted(points[mid:], key=self.by_y)))

                self.fractional_cascade(assoc, assoc_left, self.get_y, LEFT)
                self.fractional_cascade(assoc, assoc_right, self.get_y, RIGHT)

                # close modifications
                assoc = list(map(lambda y: (y[COORD_INDEX], tuple(y[POINTER_INDEX])), assoc))

                # store the max in the left subtree, the max is at loc mid - 1
                v = Node2D(None, None, points[mid - 1][self.depth], assoc)

                v.left = self.build_layered_range_tree_helper(points[:mid], assoc_left)
                v.right = self.build_layered_range_tree_helper(points[mid:], assoc_right)

                return v
        return None

    @staticmethod
    def fractional_cascade(p_assoc, assoc_child, getpoint, direction):
        """Set the pointers in the array stored at the parent. p_assoc is the y-sorted array which is
        split to two assoc_child arrays"""

        j = 0
        for i in range(len(p_assoc)):
            if getpoint(p_assoc[i]) > getpoint(assoc_child[j]):
                # try to get to the next larger value of assoc_child. if none exists break.
                j += 1
                while j < len(assoc_child) and getpoint(assoc_child[j]) == getpoint(assoc_child[j - 1]):
                    j += 1
            if j == len(assoc_child):
                break
            p_assoc[i][POINTER_INDEX][direction] = j

    @staticmethod
    def bin_search_low(A, getpoint, x: float):
        """Searches for the smallest key >= x"""
        if getpoint(A[-1]) < x:
            return -1
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

    def filter_by_y(self, node, i, y2):
        """Find the values which are less than y"""
        if i == -1:
            return []
        else:
            h = i
            n = len(node.assoc)
            sub = []
            while h < n and self.get_y(node.assoc[h]) < y2:
                sub.append(node.assoc[h][COORD_INDEX])
                h += 1
            return sub

    @staticmethod
    def trickle_down(node, i, direction):
        """ The value of i can only increase."""
        pointer = node.assoc[i][POINTER_INDEX]
        return pointer[direction]

    def get_y(self, item):
        return item[COORD_INDEX][self.depth + 1]

    def get_x(self, item):
        return item[COORD_INDEX][self.depth]

    def query_layered_range_tree(self, x1, x2, y1, y2):
        if x1 > x2:
            x1, x2 = x2, x1
        if y1 > y2:
            y1, y2 = y2, y1

        output = []

        v_split = self.find_split_node(x1, x2)
        v_i = self.bin_search_low(v_split.assoc, self.get_y, y1)
        if v_i == -1:
            return []

        i = v_i
        if self.isleaf(v_split):
            # check if the point in v_split, leaves have no pointers
            if x1 <= self.by_x(v_split.point) < x2 and y1 <= self.by_y(v_split.point) < y2:
                output.append(v_split.point)
        else:
            i = self.trickle_down(v_split, i, LEFT)
            v = v_split.left

            while not self.isleaf(v) and v is not None and i != -1:
                if v.point >= x1:
                    # report right subtree
                    a = self.trickle_down(v, i, RIGHT)
                    subset = self.filter_by_y(v.right, a, y2)
                    output += subset

                    i = self.trickle_down(v, i, LEFT)
                    v = v.left
                else:
                    i = self.trickle_down(v, i, RIGHT)
                    v = v.right

            # v is now a leaf
            if self.isleaf(v) and y1 <= self.by_y(v.point) < y2 and x1 <= self.by_x(v.point) < x2:
                output.append(v.point)
            # now we follow right side
            i = self.trickle_down(v_split, v_i, RIGHT)
            v = v_split.right
            while v is not None and not self.isleaf(v) and i != -1:
                if v.point < x2:
                    # report left subtree
                    a = self.trickle_down(v, i, LEFT)
                    subset = self.filter_by_y(v.left, a, y2)
                    output += subset

                    i = self.trickle_down(v, i, RIGHT)
                    v = v.right

                else:
                    i = self.trickle_down(v, i, LEFT)
                    v = v.left

            # check whether this point should be included too
            if v is not None and self.isleaf(v) and y1 <= self.by_y(v.point) < y2 and x1 <= self.by_x(v.point) < x2:
                output.append(v.point)

        return output


if __name__ == '__main__':
    from random import randint

    lim = 2000


    def randy():
        yield randint(0, lim)


    test_rounds = 100000
    num_coords = 1000
    x1, x2, y1, y2 = 20, 1200, 300, 400
    for _ in range(test_rounds):
        coordinates = [tuple([next(randy()), next(randy())]) for _ in range(num_coords)]
        # coordinates = [(64, 47), (37, 11), (21, 89), (41, 80), (73, 100), (26, 47)]
        r2d = LayeredRangeTree(coordinates)
        # print(r2d)
        rep = r2d.query_layered_range_tree(x1, x2, y1, y2)
        range_list = list(rep)
        brute = list(brute_algorithm(coordinates, x1, x2, y1, y2))
        print(len(brute), len(range_list))
        # print(range_list, '\n', brute)
        if len(range_list) != len(brute):
            print(coordinates)
            print(len(brute), len(range_list))
            raise ValueError()

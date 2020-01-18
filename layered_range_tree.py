from range1d import Node, Leaf
from dataclasses import dataclass
from rangetree import *
from operator import itemgetter

LEFT, RIGHT = 0, 1


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
                assoc_left = list(map(lambda w: [w, [-1, -1]], sorted(points[:mid], key=self.by_y)))
                assoc_right = list(map(lambda w: [w, [-1, -1]], sorted(points[mid:], key=self.by_y)))

                self.fractional_cascade(assoc, assoc_left, self.get_y, LEFT)
                self.fractional_cascade(assoc, assoc_right, self.get_y, RIGHT)

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

        j = 0
        for i in range(len(p_assoc)):
            if getpoint(p_assoc[i]) > getpoint(assoc_child[j]):
                j += 1
            if j == len(assoc_child):
                break
            p_assoc[i][1][direction] = j

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

    def filter_by_y(self, node, i, y2):
        """Find the values which are less than y"""
        assert i is not None
        if i >= len(node.assoc) or i is None:
            return []
        else:
            h = i
            n = len(node.assoc) - i
            sub = []
            while h < n and self.get_y(node.assoc[h]) < y2:
                sub.append(node.assoc[h][0])
                h += 1
            return sub
    
    @staticmethod
    def trickle_down(node, i, direction):
        if node is None:
            return i
        n1 = node.assoc[i]

        next_i = n1[1][direction]

        while n1[1][direction] == -1 and next_i != 0:
            i -= 1
            n1 = node.assoc[i]
            next_i = node.assoc[i][1][direction]
        return next_i

    def get_y(self, item):
        return item[0][self.depth + 1]

    def get_x(self, item):
        return item[0][self.depth]

    def query_layered_range_tree(self, x1, x2, y1, y2):
        if y1 > y2:
            y1, y2 = y2, y1

        output = []

        v_split = self.find_split_node(x1, x2)
        v_i = self.bin_search_low(v_split.assoc, self.get_y, y1)

        i = v_i
        if self.isleaf(v_split):
            # check if the point in v_split, leaves have no pointers
            if x1 <= self.by_x(v_split.point) < x2 and y1 <= self.by_y(v_split.point) < y2:
                output.append(v_split.point)
        else:
            i = self.trickle_down(v_split, i, LEFT)
            v = v_split.left

            while not self.isleaf(v):
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
            if y1 <= self.by_y(v.point) < y2 and x1 <= self.by_x(v.point) < x2:
                output.append(v.point)
            # now we follow right side
            i = self.trickle_down(v_split, v_i, RIGHT)
            v = v_split.right
            while v is not None and not self.isleaf(v):
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
            if v is not None and y1 <= self.by_y(v.point) < y2 and x1 <= self.by_x(v.point) < x2:
                output.append(v.point)
        return output


if __name__ == '__main__':
    from random import randint

    lim = 100000


    def randy():
        yield randint(0, lim)


    test_rounds = 10000
    num_coords = 70000
    x1, x2, y1, y2 = 0, 40000, 0, 50000
    for _ in range(test_rounds):
        coordinates = [tuple([next(randy()), next(randy())]) for _ in range(num_coords)]
        r2d = LayeredRangeTree(coordinates)
        # print(r2d)
        rep = r2d.query_layered_range_tree(x1, x2, y1, y2)
        range_list = list(rep)
        brute = list(brute_algorithm(coordinates, x1, x2, y1, y2))
        print(len(brute), len(range_list))
        # print(range_list, '\n', brute)
        if len(range_list) != len(brute):
            print(len(brute), len(range_list))
            print(sorted(coordinates))
            raise ValueError()

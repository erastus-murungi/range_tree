# d-D range tree implementation
from dataclasses import dataclass
from operator import itemgetter
from typing import Union

from layered_range_tree import LayeredRangeTree
from range1d import Leaf, Node
from rangetree import RangeTree


@dataclass
class NodeDD(Node):
    next_tree: Union[LayeredRangeTree, "DDRangeTree"]


@dataclass
class LeafDD(Leaf):
    next_tree: Union[LayeredRangeTree, "DDRangeTree"]


class DDRangeTree(RangeTree):
    """Uses a Layered Range tree for the base case."""

    def __init__(self, points, depth=0):
        super().__init__()
        # self.max_depth is constant at every level since we are not modifying the list size
        self.max_depth = len(points[0])
        self.root = self.build_dd_range_tree(points, depth)

    @staticmethod
    def is_leaf(node):
        return type(node) == Leaf or type(node) == LeafDD

    def build_dd_range_tree(self, points, axis):
        sorted_points = sorted(points, key=itemgetter(axis))
        return self.build_dd_range_tree_helper(sorted_points, axis)

    def build_dd_range_tree_helper(self, points, depth):
        if points:
            if self.max_depth - depth == 3:
                tree = LayeredRangeTree(points, depth + 1)  # range 2D can be used here
            else:
                tree = DDRangeTree(points, depth + 1)
            if len(points) == 1:
                return LeafDD(points[0], tree)
            else:
                mid = (len(points)) >> 1
                v = NodeDD(None, None, points[mid - 1][depth], tree)
                v.right = self.build_dd_range_tree_helper(points[mid:], depth)
                v.left = self.build_dd_range_tree_helper(points[:mid], depth)
                return v
        return None

    @staticmethod
    def qualifies(node, queries):
        """Checks if a leaf node should be reported."""
        # assert self.isleaf(node)
        for axis, (i, j) in enumerate(queries):
            if not (i <= node.point[axis] < j):
                return False
        return True

    def __filter(self, node, curr_depth, queries):
        """Determines which type of method to call in the next level of recursion."""
        if self.max_depth - curr_depth == 3:
            return node.next_tree.query_layered_range_tree(
                *queries[curr_depth + 1], *queries[curr_depth + 2]
            )
        else:
            return node.next_tree.query_dd_range_tree(queries, curr_depth + 1)

    def query_dd_range_tree(self, queries, axis=0):
        """Takes as arguments a tuples of coordinates.
        very similar to range1D query"""
        assert len(queries) == self.max_depth
        i, j = queries[axis]

        if i > j:
            i, j = j, i

        output = []
        v_split = self.find_split_node(i, j)
        if self.is_leaf(v_split):
            # check if the point in v_split
            if self.qualifies(v_split, queries):
                output.append(v_split)
        else:
            v = v_split.left
            while not self.is_leaf(v):
                if v.point >= i:
                    # report right subtree
                    subset = self.__filter(v.right, axis, queries)
                    output += subset
                    v = v.left
                else:
                    v = v.right
            # v is now a leaf
            if self.qualifies(v, queries):
                output.append(v)
            # now we follow right side
            v = v_split.right
            while v is not None and not self.is_leaf(v):
                if v.point < j:
                    # report left subtree
                    subset = self.__filter(v.left, axis, queries)
                    output += subset
                    # it is possible to traverse to an external node
                    v = v.right
                else:
                    v = v.left
            # check whether this point should be included too
            if v is not None and self.qualifies(v, queries):
                output.append(v)
        return output


def brute(ps, qs):
    for p in ps:
        for k, (i, j) in enumerate(qs):
            if not (i <= p[k] < j):
                break
            if k + 1 == len(qs):
                yield p


if __name__ == "__main__":
    from datetime import datetime
    from random import randint

    from pympler import asizeof

    lim = 100
    d = 5
    num_coords = 300
    test_rounds = 1

    def randy():
        yield randint(0, lim)

    q = [(30, 100), (10, 100), (30, 80), (20, 80), (45, 76)]
    for i in range(test_rounds):
        coordinates = [
            tuple([next(randy()) for _ in range(d)]) for _ in range(num_coords)
        ]
        t1 = datetime.now()
        rdd = DDRangeTree(coordinates)
        print(
            "This object uses:",
            f"{asizeof.asizeof(rdd) / (2 ** 20):.3f}",
            "MB and is constructed in:",
            (datetime.now() - t1).total_seconds(),
            "seconds.",
        )

        t2 = datetime.now()
        print(
            "Brute algorithm query ran in:",
            (datetime.now() - t2).total_seconds(),
            "seconds",
        )
        nb = len(list(brute(coordinates, q)))

        t3 = datetime.now()
        rep = rdd.query_dd_range_tree(q)
        nr = len(list(rdd.report(rep)))
        print(
            "d-D range query ran in:", (datetime.now() - t3).total_seconds(), "seconds"
        )

        print(nr, nb)
        if nr != nb:
            print(coordinates)
            raise ValueError()

        __output__ = """ This object uses: 274.593 MB and is constructed in: 26.46904 seconds.
                        Brute algorithm query ran in: 1e-06 seconds
                        d-D range query ran in: 0.000511 seconds
                        19 19"""

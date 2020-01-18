# d-D range tree implementation
from layered_range_tree import LayeredRangeTree
from typing import Union
from rangetree import RangeTree
from dataclasses import dataclass
from operator import itemgetter
from range1d import Node, Leaf


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
    def isleaf(node):
        return type(node) == Leaf or type(node) == LeafDD

    def build_dd_range_tree(self, points, axis):
        points.sort(key=itemgetter(axis))  # sort by first element
        return self.build_dd_range_tree_helper(points, axis)

    def build_dd_range_tree_helper(self, points, depth):
        if points:
            if self.max_depth - depth == 3:
                tree = LayeredRangeTree(points, depth + 1)  # range 2D can be used here
            else:
                tree = DDRangeTree(points, depth + 1)
            if len(points) == 1:
                v = LeafDD(points[0], tree)
                return v
            else:
                mid = (len(points)) >> 1
                v = NodeDD(None, None,
                           points[mid-1][depth], tree)
                v.right = self.build_dd_range_tree_helper(points[mid:], depth)
                v.left = self.build_dd_range_tree_helper(points[:mid], depth)
                return v
        return None

    def qualifies(self, node, queries):
        assert self.isleaf(node)
        for axis, (i, j) in enumerate(queries):
            if not (i < node.point[axis] < j):
                return False
        return True

    def __filter(self, node, curr_depth, queries):
        if self.max_depth - curr_depth == 3:
            return node.next_tree.query_layered_range_tree(*queries[curr_depth + 1], *queries[curr_depth + 2])
        else:
            return node.next_tree.query_dd_range_tree(self, curr_depth + 1, queries)

    def query_dd_range_tree(self, queries, axis=0):
        """Takes as arguments a tuples of coordinates"""
        assert len(queries) == self.max_depth
        i, j = queries[axis]

        if i > j:
            i, j = j, i

        output = []
        v_split = self.find_split_node(i, j)
        if self.isleaf(v_split):
            # check if the point in v_split
            if self.qualifies(v_split, queries):
                output.append(v_split)
        else:
            v = v_split.left
            while not self.isleaf(v):
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
            while v is not None and not self.isleaf(v):
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
            if not(i <= p[k] < j):
                break
            if k + 1 == len(qs):
                yield p


if __name__ == '__main__':
    from random import randint
    from pprint import pprint

    lim = 20
    d = 3
    num_coords = 9

    def randy():
        yield randint(0, lim)
    q = [(5, 15), (2, 19), (3, 10)]
    # coordinates = [tuple([next(randy()) for _ in range(d)]) for _ in range(num_coords)]
    coordinates = [(12, 10, 10), (9, 7, 4), (7, 8, 3), (13, 10, 20), (1, 19, 10), (19, 5, 9), (12, 0, 10), (13, 16, 0), (3, 9, 16)]

    print(coordinates)
    rdd = DDRangeTree(coordinates)

    print(list(brute(coordinates, q)))
    rep = rdd.query_dd_range_tree(q)
    pprint(list(rdd.report(rep)))

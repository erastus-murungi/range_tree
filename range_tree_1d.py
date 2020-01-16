from dataclasses import dataclass
from typing import Union, Tuple, Callable, List
from sys import maxsize
from operator import itemgetter
from collections.abc import Iterable
from ptree import printtree


@dataclass
class Leaf:
    point: Union[Tuple, float]
    parent: Union["Node", None]


@dataclass
class Node:
    left: Union[Leaf, "Node", None]
    right: Union[Leaf, "Node", None]
    point: float
    parent: Union["Node", None]


class RangeTree1D:
    """A 1D Range Tree."""
    INF = maxsize

    def __init__(self, values, axis=0):
        self.root, self.levels = self.build_range_tree(values, axis)

    @staticmethod
    def isleaf(node) -> bool:
        return type(node) == Leaf

    def height(self, node) -> int:
        return -1 if node is None else 0 if self.isleaf(node) else max(self.height(node.left),
                                                                       self.height(node.right)) + 1

    def split_value(self, node, get: Callable) -> float:
        """This is just the maximum value in the left subtree"""

        if node is None:
            return 0
        elif self.isleaf(node):
            return get(node.point)
        else:
            return max(node.point, self.split_value(node.left, get), self.split_value(node.right, get))

    def build_range_tree(self, values, axis=0) -> Tuple[Union[Leaf, Node], List]:
        """ Build a 1D Range Tree from the bottom up and returns the root, and the nodes on the same level.
            This is just for indexing.
            It is possible to augment the structure to store any information.
        """

        if not values:
            raise ValueError("Empty iterable")
        if len(values) == 1:
            levels = [[Leaf(values[0], None)]]
            return levels[-1][0], levels
        getter = itemgetter(axis) if isinstance(values[0], Iterable) else lambda y: y

        # O(n log n) because of sorting
        leaves = list(map(lambda val: Leaf(val, None), sorted(values, key=getter)))
        levels = [leaves]
        # n + n/2 + n/4 + n/8 + ... + 1 ≅ 2n  (Geometric summation) = O(n)
        while (n := len(leaves)) > 1:
            nodes = []
            for i in range(1, n, 2):
                l, r = leaves[i - 1], leaves[i]
                x = Node(l, r, self.split_value(l, getter), None)
                l.parent = r.parent = x
                nodes.append(x)
            if n & 1:  # if odd
                x = Node(leaves[n - 1], None, self.split_value(leaves[n - 1], getter), None)
                nodes.append(x)
                leaves[n - 1].parent = x
            leaves = nodes
            levels.append(leaves)

        # Total running time is: O(n log n)
        return levels[-1][0], levels

    def report(self, output: list):
        """Generates the nodes in the subtrees in the order in which they were found."""

        def __report_helper(n):
            if n is not None:
                if self.isleaf(n.left):
                    yield n.left.point
                else:
                    yield from __report_helper(n.left)
                if self.isleaf(n.right):
                    yield n.right.point
                else:
                    yield from __report_helper(n.right)

        for node in output:
            if self.isleaf(node):
                yield node.point
            else:
                yield from __report_helper(node)

    def find_split_node(self, x, y) -> Node:
        """ Finds and returns the split node
            For the range query [x : x'], the node v in a balanced binary search
            tree is a split node if its value x.v satisfies x.v ≥ x and x.v < x'.
        """

        v = self.root
        while not self.isleaf(v) and (v.point >= y or v.point < x):
            v = v.left if y <= v.point else v.right
        return v

    def query_range_tree(self, i, j) -> List:
        """ Queries a 1D Range Tree.

            Let P be a set of n points in 1-D space. The set P
            can be stored in a balanced binary search tree, which uses O(n) storage and
            has O(n log n) construction time, such that the points in a query range can be
            reported in time O(k + log n), where k is the number of reported points.
        """

        if i > j:
            i, j = j, i

        output = []
        v_split = self.find_split_node(i, j)
        if self.isleaf(v_split):
            # check if the point in v_split
            if i <= v_split.point <= j:  # inclusive version
                output.append(v_split)
        else:
            v = v_split.left
            while not self.isleaf(v):
                if v.point >= i:
                    # report right subtree
                    output.append(v.right)
                    v = v.left
                else:
                    v = v.right
            # v is now a leaf
            if v.point >= i:
                output.append(v)
            # now we follow right side
            v = v_split.right
            while not self.isleaf(v):
                if v.point < j:
                    # report left subtree
                    output.append(v.left)
                    # it is possible to traverse to an external node
                    if v.right is None:
                        return output
                    v = v.right
                else:
                    v = v.left
            # check whether this point should be included too
            if v.point < j:
                output.append(v)
        return output

    def __repr__(self) -> str:
        return repr(self.root)

    def __getitem__(self, item: slice):
        """Assumes item is a slice object.
        To search for a specific value:
        Use that value in both endpoints. eg to search for 5, query [5:5].
        Returns the items in the range.
        """
        assert isinstance(item, slice), print(item)

        start, stop = item.start, item.stop
        if start is None:
            start = 0
        if stop is None:
            stop = self.INF
        if start > stop:
            raise IndexError("make sure start <= stop")

        return self.query_range_tree(start, stop)


if __name__ == '__main__':
    points = [4, 6]
    rtree = RangeTree1D(points)
    rep = rtree[:]
    gen = list(rtree.report(rep))
    printtree(rtree)

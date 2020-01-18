from dataclasses import dataclass
from typing import Union, Tuple, List
from sys import maxsize
from operator import itemgetter
from collections.abc import Iterable
from rangetree import RangeTree


@dataclass
class Leaf:
    point: Union[Tuple, float]


@dataclass
class Node:
    left: Union[Leaf, "Node", None]
    right: Union[Leaf, "Node", None]
    point: float


class RangeTree1D(RangeTree):
    """A 1D Range Tree."""
    INF = maxsize

    def __init__(self, values, axis=0):
        super().__init__()
        self.root, self.levels = self.build_range_tree(values, axis)

    @staticmethod
    def isleaf(node) -> bool:
        return type(node) == Leaf

    def build_range_tree(self, values, axis) -> Tuple[Union[Leaf, Node], List]:
        """ Build a 1D Range Tree from the bottom up and returns the root, and the nodes on the same level.
            This is just for indexing.
            It is possible to augment the structure to store any information.
        """

        if not values:
            raise ValueError("Empty iterable")
        if len(values) == 1:
            levels = [[Leaf(values[0])]]
            return levels[-1][0], levels
        getter = itemgetter(axis) if isinstance(values[0], Iterable) else lambda y: y

        # O(n log n) because of sorting
        leaves = list(map(lambda val: Leaf(val), sorted(values, key=getter)))
        levels = [leaves]
        # n + n/2 + n/4 + n/8 + ... + 1 ≅ 2n  (Geometric summation) = O(n)
        while (n := len(leaves)) > 1:
            nodes = []
            for i in range(1, n, 2):
                l, r = leaves[i - 1], leaves[i]
                nodes.append(Node(l, r, self.split_value(l, getter)))
            if n & 1:  # if odd
                nodes.append(Node(leaves[n - 1], None, self.split_value(leaves[n - 1], getter)))
            leaves = nodes
            levels.append(leaves)

        # Total running time is: O(n log n)
        return levels[-1][0], levels

    def query_range_tree1d(self, i, j, get_point=lambda w: w.point) -> List:
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
            if i <= get_point(v_split) < j:  # inclusive version
                output.append(v_split)
        else:
            v = v_split.left
            while not self.isleaf(v):
                if get_point(v) >= i:
                    # report right subtree
                    output.append(v.right)
                    v = v.left
                else:
                    v = v.right
            # v is now a leaf
            if i <= get_point(v) < j:
                output.append(v)
            # now we follow right side
            v = v_split.right
            while not self.isleaf(v):
                if get_point(v) < j:
                    # report left subtree
                    output.append(v.left)
                    # it is possible to traverse to an external node
                    if v.right is None:
                        return output
                    v = v.right
                else:
                    v = v.left
            # check whether this point should be included too
            if i <= get_point(v) < j:
                output.append(v)
        return output

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

        return self.query_range_tree1d(start, stop)


def brute(points, x1, x2):
    for point in points:
        if x1 <= point < x2:
            yield point


if __name__ == '__main__':
    from random import randint

    lim = 1000

    def randy():
        yield randint(0, lim)

    num_rounds = 10000
    num_points = 100
    x1, x2 = 400, 600

    for _ in range(num_rounds):
        points = [next(randy()) for _ in range(num_points)]
        rtree = RangeTree1D(points)

        m = len(list(brute(points, x1, x2)))
        rep = rtree[x1:x2]
        n = len(list(rtree.report(rep)))
        if n != m:
            raise ValueError()

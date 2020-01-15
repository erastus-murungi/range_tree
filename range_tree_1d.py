from dataclasses import dataclass
from typing import Union
from sys import maxsize


@dataclass
class Leaf:
    key: float
    parent: Union["Node", None] = None


@dataclass
class Node:
    left: Union[Leaf, "Node", None]
    right: Union[Leaf, "Node", None]
    key: float
    parent: Union["Node", None] = None


class RangeTree1D:
    """A 1D Range Tree."""
    INF = maxsize

    def __init__(self, values):
        self.root, self.levels = self.build_range_tree(values)

    @staticmethod
    def isleaf(node):
        return type(node) == Leaf

    def height(self, node):
        return -1 if node is None else 0 if self.isleaf(node) else max(self.height(node.left),
                                                                       self.height(node.right)) + 1

    def split_value(self, node):
        """This is just the maximum value in the left subtree"""

        return 0 if node is None \
            else node.key if self.isleaf(node) \
            else max(node.key, self.split_value(node.left), self.split_value(node.right))

    def build_range_tree(self, values):
        """ Build a 1D Range Tree and returns the root, and the nodes on the same level
            This is just for indexing.
            It is possible to augment the structure to store any information."""

        if not values:
            raise ValueError("Empty iterable")

        # O(n log n) because of sorting
        leaves = list(map(lambda val: Leaf(val, None), sorted(values)))
        if len(leaves) == 1:
            return Leaf(values[0], None)

        levels = [leaves]
        # n + n/2 + n/4 + n/8 + ... + 1 ≅ 2n  (Geometric summation) = O(n)
        while (n := len(leaves)) > 1:
            nodes = []
            for i in range(1, n, 2):
                l, r = leaves[i - 1], leaves[i]
                x = Node(l, r, self.split_value(l), None)
                l.parent = r.parent = x
                nodes.append(x)
            if n & 1:  # if odd
                x = Node(leaves[n - 1], None, self.split_value(leaves[n - 1]), None)
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
                    yield n.left.key
                else:
                    yield from __report_helper(n.left)
                if self.isleaf(n.right):
                    yield n.right.key
                else:
                    yield from __report_helper(n.right)

        for node in output:
            if self.isleaf(node):
                yield node.key
            else:
                yield from __report_helper(node)

    def find_split_node(self, x, y):
        """ Finds and returns the split node
            For the range query [x : x'], the node v in a balanced binary search
            tree is a split node if its value x.v satisfies x.v ≥ x and x.v < x'

            FIND_SPLIT_NODE(T,x,x)
                Input: A tree T and two values x and x' with x < x'.
                Outpu: The node ν where the paths to x and x split, or the leaf where both paths end.
                1. ν <- root(T)
                2. while ν is not a leaf and (x'<= x_v.key or x > x_v.key )
                3.      do if x'<=x_v.key
                4.      then ν <- leftchild(ν)
                5.      else ν <- rightchild(ν)
                6. return ν
        """

        v = self.root
        while not self.isleaf(v) and (v.key >= y or v.key < x):
            v = v.left if y <= v.key else v.right
        return v

    def query_range_tree(self, i, j):
        """ Queries a 1D Range Tree.

            Let P be a set of n points in 1-D space. The set P
            can be stored in a balanced binary search tree, which uses O(n) storage and
            has O(n log n) construction time, such that the points in a query range can be
            reported in time O(k + log n), where k is the number of reported points"""

        if i > j:
            i, j = j, i

        output = []
        v_split = self.find_split_node(i, j)
        if self.isleaf(v_split):
            # check if the key in v_split
            if i <= v_split.key <= j:  # inclusive version
                output.append(v_split)
        else:
            v = v_split.left
            while not self.isleaf(v):
                if i <= v.key:
                    # report right subtree
                    output.append(v.right)
                    v = v.left
                else:
                    v = v.right
            # v is now a leaf
            if v.key >= i:
                output.append(v)
            # now we follow right side
            v = v_split.right
            while not self.isleaf(v):
                if v.key < j:
                    # report left subtree
                    output.append(v.left)
                    # it is possible to traverse to an external node
                    if v.right is None:
                        return output
                    v = v.right
                else:
                    v = v.left
        return output

    def __getitem__(self, item: slice):
        """Assumes item is a slice object
        Returns the items in the range
        """
        assert isinstance(item, slice), print(item)

        start, stop = item.start, item.stop
        if stop is None:
            stop = self.INF
        if start > stop:
            raise IndexError("make sure start <= stop")

        return self.query_range_tree(start, stop)


if __name__ == '__main__':
    points = [3, 10, 19, 23, 30, 37, 45, 59, 62, 70, 80, 89, 100, 105]
    rtree = RangeTree1D(points)
    # vals = [list(map(lambda x: x.key, values)) for values in levels_]
    # vals.reverse()
    # pprint(vals)

    rep = rtree[0: 120]
    gen = list(rtree.report(rep))
    print(gen)

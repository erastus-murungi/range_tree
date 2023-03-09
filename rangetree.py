from abc import ABC
from functools import cache
from sys import maxsize
from typing import Union, NamedTuple, Iterator

import numpy as np


class Leaf(NamedTuple):
    point: np.ndarray

    def __hash__(self):
        return id(self)


OUT_OF_BOUNDS: Leaf = Leaf(np.array([-maxsize]))


class Split:
    __slots__ = ("split_value", "left", "right")

    def __init__(
        self,
        split_value: float,
        left: Union["Split", Leaf],
        right: Union["Split", Leaf],
    ):
        self.split_value = split_value
        self.left = left
        self.right = right

    def is_leaf(self):
        return self.left is None and self.right is None

    def __repr__(self):
        return f"{type(self).__name__}({self.split_value})"


Node = Split | Leaf


class RangeTree(ABC):
    def __init__(self, root: Node):
        self.root: Node = root

    @cache
    def split_value(self, node: Node, axis: int) -> float:
        """This is just the maximum value in the left subtree"""
        if isinstance(node, Leaf):
            if node is OUT_OF_BOUNDS:
                return node.point[0]
            return node.point[axis]
        else:
            return max(
                node.split_value,
                self.split_value(node.left, axis),
                self.split_value(node.right, axis),
            )

    def report_leaves(self, n: Node):
        if isinstance(n, Leaf):
            if n is not OUT_OF_BOUNDS:
                yield n.point
        else:
            yield from self.report_leaves(n.left)
            yield from self.report_leaves(n.right)

    def find_split_node(self, x, y):
        """Finds and returns the split node
        For the range query [x : x'], the node v in a balanced binary search
        tree is a split node if its value x.v satisfies x.v ≥ x and x.v < x'.
        """

        v = self.root
        while not isinstance(v, Leaf) and (v.split_value >= y or v.split_value < x):
            v = v.left if y <= v.split_value else v.right
        return v

    def pretty_print_tree(self):
        def print_nodes(node: Node, indent: str, prefix: str) -> Iterator[str]:
            """Prints the nodes of the tree line by line"""
            yield f"{indent}{prefix}----{node}\n"
            if isinstance(node, Split):
                indent += "     " if prefix == "R" else "|    "
                yield from print_nodes(node.left, indent, "L")
                yield from print_nodes(node.right, indent, "R")

        return "".join(print_nodes(self.root, "", "R"))

    def __repr__(self) -> str:
        return repr(self.root)

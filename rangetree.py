from abc import ABC, abstractmethod
from sys import maxsize
from typing import Iterator

import numpy as np

from utils import Interval, Orthotope


class RangeTree(ABC):
    @abstractmethod
    def find_split_value(self) -> float:
        pass

    @abstractmethod
    def report_leaves(self):
        pass

    @staticmethod
    @abstractmethod
    def construct(self, axis):
        pass

    @abstractmethod
    def yield_line(self, indent: str, prefix: str) -> Iterator[str]:
        pass

    @abstractmethod
    def query(self, item, axis: int = 0):
        pass

    @property
    def assoc(self):
        return self

    @staticmethod
    def find_split_node(root, interval: Interval):
        """Finds and returns the split node
        For the range query [x : x'], the node v in a balanced binary search
        tree is a split node if its value x.v satisfies x.v ≥ x and x.v < x'.
        """

        v = root
        while not isinstance(v, Leaf) and (
            interval.end <= v.split_value or interval.start > v.split_value
        ):
            v = v.left if interval.end <= v.split_value else v.right
        return v

    def pretty_str(self):
        return "".join(self.yield_line("", "R"))


class Leaf(RangeTree):
    __slots__ = ("point", "axis")

    def __init__(self, point, axis: int):
        self.point = point
        self.axis = axis

    construct = __init__

    def query(self, bound: Orthotope, axis: int = 0):
        if self is not OUT_OF_BOUNDS and self.point[axis:] in bound[axis:]:
            yield self.point

    def query_interval(self, interval: Interval, axis: int = 0):
        if self is not OUT_OF_BOUNDS and self.point[axis] in interval:
            yield self.point

    @property
    def split_value(self):
        return self.point[self.axis]

    def find_split_value(self) -> float:
        return self.split_value

    def report_leaves(self):
        if self is not OUT_OF_BOUNDS:
            yield self.point

    def yield_line(self, indent: str, prefix: str) -> Iterator[str]:
        yield f"{indent}{prefix}----{self}\n"

    def __hash__(self):
        return id(self)

    def __repr__(self):
        return f"Leaf({self.point})"


OUT_OF_BOUNDS: Leaf = Leaf(np.array([-maxsize]), 0)

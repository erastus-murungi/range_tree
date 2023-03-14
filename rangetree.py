from abc import ABC, abstractmethod
from sys import maxsize
from typing import Collection, Iterator, NamedTuple

import numpy as np


class Interval(NamedTuple):
    start: float
    end: float

    def __contains__(self, item: float):
        return self.start <= item < self.end

    def is_disjoint_from(self, other: "Interval"):
        return self.start > other.end or other.start > self.end

    def contains(self, other: "Interval"):
        return other.start >= self.start and other.end < self.end

    def copy(self):
        return Interval(self.start, self.end)


class Orthotope(NamedTuple):
    intervals: list[Interval]

    def __contains__(self, points: Collection[float]):
        if len(points) != len(self.intervals):
            raise ValueError(
                "expected the number of intervals to equal the number of points"
            )
        return all(value in interval for value, interval in zip(points, self.intervals))

    def __iter__(self):
        yield from self.intervals

    @property
    def x_range(self):
        return self.intervals[0]

    @property
    def y_range(self):
        return self.intervals[1]

    def __getitem__(self, item):
        return Orthotope(self.intervals[item.start :])

    def copy(self):
        return Orthotope([interval.copy() for interval in self.intervals])

    def split(self, dim: int, split_value: float) -> tuple["Orthotope", "Orthotope"]:
        region_right = self.copy()
        dim_interval = self.intervals[dim]
        if dim_interval.start <= split_value <= dim_interval.end:
            self.intervals[dim] = Interval(dim_interval.start, split_value)
            region_right.intervals[dim] = Interval(split_value, dim_interval.end)
            return self, region_right
        raise ValueError()

    def contains(self, other: "Orthotope"):
        return all(
            my_interval.contains(other_interval)
            for my_interval, other_interval in zip(self.intervals, other.intervals)
        )

    def is_disjoint_from(self, other: "Orthotope"):
        return all(
            my_interval.is_disjoint_from(other_interval)
            for my_interval, other_interval in zip(self.intervals, other.intervals)
        )

    def __len__(self):
        return len(self.intervals)


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

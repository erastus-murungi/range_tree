from bisect import insort
from operator import attrgetter
from sys import maxsize
from typing import Callable, NamedTuple

import numpy as np
import numpy.typing as npt


def l2_norm(a, b):
    return np.linalg.norm(a - b)


Point = npt.NDArray[float]
Points = Point


class NNResult(NamedTuple):
    point: Point
    distance: float


class BoundedPriorityQueue(list[NNResult]):
    """Fast list-based bounded priority queue."""

    __slots__ = ("_capacity", "_distance_function", "_reference_point")

    def __init__(
        self,
        capacity: int,
        reference_point: Point,
        distance_function: Callable[[Point, Point], float],
    ):
        super().__init__()
        self._reference_point = reference_point
        self._capacity = capacity
        self._distance_function = distance_function

    def append(self, point: Point, distance=None):
        if not distance:
            distance = self._distance_function(point, self._reference_point)
        if len(self) < self._capacity or distance < self[-1].distance:
            insort(
                self,
                NNResult(point, distance),
                key=attrgetter("distance"),
            )
            if len(self) > self._capacity:
                self.pop()

    def extend(self, points: Points):
        for point in points:
            self.append(point)

    def is_full(self):
        return len(self) == self._capacity

    def peek(self):
        if len(self) > 0:
            return self[-1]


def brute_nearest_neighbor(coords, query_point, distance_function):
    # naive nearest neighbor
    best_dist, best_point = maxsize, None
    for coord in coords:
        dist = distance_function(coord, query_point)
        if dist < best_dist:
            best_dist, best_point = dist, coord
    return best_point


def brute_k_nearest_neighbors(coords, query_point, k, distance_function):
    """Simple kNN for benchmarking"""
    bpq = []
    for coord in coords:
        dist = distance_function(coord, query_point)
        if len(bpq) < k or dist < bpq[-1].distance:
            insort(bpq, NNResult(coord, dist), key=attrgetter("distance"))
            if len(bpq) > k:
                bpq.pop()
    return bpq


def brute_range_search(coords, orthotope):
    for coord in coords:
        if all(x in interval for x, interval in zip(coord, orthotope)):
            yield tuple(coord)


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

    def __contains__(self, points: Point):
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

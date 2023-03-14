from bisect import insort
from typing import NamedTuple

import numpy as np

from rbt import RedBlackTree


class NNResult(NamedTuple):
    point: np.ndarray
    distance: float


class BoundedPriorityQueueRBT:
    """Simple bounded priority queue which uses a Red Black Tree as the underlying data structure."""

    def __init__(self, k):
        self.maxkey = -float("inf")
        if k < 0:
            raise ValueError("k should be larger than 0")
        self.k = k
        self._bpq = RedBlackTree()

    def insert(self, key, item):
        if self.k == 0:
            return self.maxkey
        if len(self._bpq) < self.k or key < self.maxkey:
            self._bpq.insert(key, item)  # O (lg n)
        if len(self._bpq) > self.k:
            self._bpq.delete(self._bpq.maximum[0])  # O(lg n)
        self.maxkey = self._bpq.maximum[0]  # (lg n)

    def __setitem__(self, key, item=0):
        assert (type(key)) in [int, float], "Invalid type of key."
        self.insert(key, item)

    @property
    def isfull(self):
        return len(self._bpq) == self.k

    def iteritems(self):
        return self._bpq.iteritems()

    def __repr__(self):
        return repr(self._bpq.root)

    def __str__(self):
        return str(list(self._bpq.iteritems()))


class BoundedPriorityQueue(list[NNResult]):
    """Fast list-based bounded priority queue."""

    __slots__ = ("_capacity", "_distance_function", "_base")

    def __init__(self, capacity: int, base, distance_function):
        super().__init__()
        self._base = base
        self._capacity = capacity
        self._distance_function = distance_function

    def append(self, point: np.ndarray):
        distance = self._distance_function(point, self._base)
        if len(self) < self._capacity or distance < self[-1].distance:
            insort(
                self,
                NNResult(point, distance),
                key=lambda nn_result: nn_result.distance,
            )
            if len(self) > self._capacity:
                self.pop()

    def extend(self, points):
        for point in points:
            self.append(point)

    def is_full(self):
        return len(self) == self._capacity

    def peek(self):
        if len(self) > 0:
            return self[-1]

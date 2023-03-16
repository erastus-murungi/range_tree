from abc import ABC, abstractmethod
from bisect import insort
from operator import attrgetter, itemgetter
from sys import maxsize
from typing import Callable, Iterator, NamedTuple, Optional

import numpy as np
import numpy.typing as npt

from rangetree import Interval, Orthotope

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


class KDNode(ABC):
    __slots__ = "data"

    def __init__(self, data: Point):
        self.data = data

    @abstractmethod
    def min(self, dim: int, cd: int, n_dims: int):
        pass

    @abstractmethod
    def max(self, dim: int, cd: int, n_dims: int):
        pass

    @abstractmethod
    def yield_line(self, indent: str, prefix: str) -> Iterator[str]:
        pass

    @abstractmethod
    def range_search(self, query: Orthotope, region: Orthotope, cd: int, n_dims: int):
        pass

    @abstractmethod
    def report_nodes(self):
        pass

    def __repr__(self):
        return f"{self.__class__.__name__}({self.data})"


class Leaf(KDNode):
    def __init__(self, data: np.ndarray):
        super().__init__(data)

    def min(self, dim: int, cd: int, n_dim: int):
        if self.data.size == 0:
            return np.full(n_dim, maxsize)
        return min(self.data, key=itemgetter(dim))

    def max(self, dim: int, cd: int, n_dim: int):
        if self.data.size == 0:
            return np.full(n_dim, -maxsize)
        return max(self.data, key=itemgetter(dim))

    def yield_line(self, indent: str, prefix: str) -> Iterator[str]:
        yield f"{indent}{prefix}----{self}\n"

    def range_search(
        self, orthotope: Orthotope, region: Orthotope, cd: int, n_dims: int
    ):
        yield from filter(lambda data: data in orthotope, self.data)

    def report_nodes(self):
        yield from self.data

    def __bool__(self):
        return bool(self.data.size)


def l2_norm(a, b):
    return np.linalg.norm(a - b)


class InternalNode(KDNode):
    __slots__ = ("left", "right")

    def __init__(self, data: np.ndarray, left: KDNode, right: KDNode):
        super().__init__(data)
        self.left = left
        self.right = right

    def min(self, dim: int, cd: int, n_dims: int):
        # T splits on the dimension we’re searching
        # => only visit left subtree
        base = min(self.data, key=itemgetter(dim))
        if cd == dim:
            return min(
                base, self.left.min(dim, (cd + 1) % n_dims, n_dims), key=itemgetter(dim)
            )
        # T splits on a different dimension
        # => have to search both subtrees
        else:
            return min(
                self.left.min(dim, (cd + 1) % n_dims, n_dims),
                self.right.min(dim, (cd + 1) % n_dims, n_dims),
                base,
                key=itemgetter(dim),
            )

    def max(self, dim: int, cd: int, n_dims: int):
        # T splits on the dimension we’re searching
        # => only visit right subtree
        base = max(self.data, key=itemgetter(dim))
        if cd == dim:
            return max(
                base,
                self.right.max(dim, (cd + 1) % n_dims, n_dims),
                key=itemgetter(dim),
            )
        # T splits on a different dimension
        # => have to search both subtrees
        else:
            return max(
                self.left.max(dim, (cd + 1) % n_dims, n_dims),
                self.right.max(dim, (cd + 1) % n_dims, n_dims),
                base,
                key=itemgetter(dim),
            )

    def yield_line(self, indent: str, prefix: str) -> Iterator[str]:
        yield f"{indent}{prefix}----{self}\n"
        indent += "     " if prefix == "R" else "|    "
        yield from self.left.yield_line(indent, "L")
        yield from self.right.yield_line(indent, "R")

    def range_search(self, query: Orthotope, region: Orthotope, cd: int, n_dims: int):
        if region.is_disjoint_from(query):
            return
        if query.contains(region):
            yield from self.report_nodes()
            return

        yield from filter(lambda point: point in query, self.data)

        next_cd = (cd + 1) % n_dims

        region_left, region_right = region.split(cd, self.data[0, cd])
        yield from self.left.range_search(query, region_left, next_cd, n_dims)
        yield from self.right.range_search(query, region_right, next_cd, n_dims)

    def report_nodes(self):
        yield from self.left.report_nodes()
        yield from self.data
        yield from self.right.report_nodes()


class KDTree:
    __slots__ = ("_size", "_n_dims", "_root", "_region", "_leaf_size", "_dist_fn")

    def __init__(
        self,
        *,
        data_points: Points = None,
        n_dims: Optional[int] = None,
        dist_fn=l2_norm,
        leaf_size: int = 1,
    ):
        """The tree can be initialized with or without points"""
        self._leaf_size = leaf_size
        self._dist_fn = dist_fn

        if n_dims is not None:
            self._size, self._n_dims = 0, n_dims
            self._root = Leaf(np.empty((0, n_dims)))
            self._region = Orthotope(
                [Interval(-maxsize, maxsize) for _ in range(n_dims)]
            )
        else:
            self._size, self._n_dims = data_points.shape
            self._root = self.build(data_points, 0, self._n_dims)
            # # calculate the size of the region
            self._region = Orthotope(
                [
                    Interval(*min_max)
                    for min_max in zip(
                        np.min(data_points, axis=0), np.max(data_points, axis=0)
                    )
                ]
            )  # O(k lg n)

    def build(self, data_points: Points, dim: int, n_dims: int) -> KDNode:
        # Order O(n^2 lg n) build time
        # Can be sped up using a linear time median_low finding algorithm
        # Or maintaining a k sorted lists of points to enable quick median finding. The construction of time f such
        # an algorithm is O(k nlg n), where k is the dimension

        # if all points are the same, just put them in the same leaf node
        if len(data_points) <= self._leaf_size or np.all(
            data_points[0, dim] == data_points[:, dim]
        ):
            return Leaf(data_points)

        data_points = data_points[data_points[:, dim].argsort()]
        split_value = data_points[(len(data_points) - 1) // 2, dim]
        return InternalNode(
            data_points[data_points[:, dim] == split_value],
            self.build(
                data_points[data_points[:, dim] < split_value],
                (dim + 1) % n_dims,
                n_dims,
            ),
            self.build(
                data_points[data_points[:, dim] > split_value],
                (dim + 1) % n_dims,
                n_dims,
            ),
        )

    def insert(self, point):
        """Inserts a single point into the KD-Tree. It does not allow duplicates"""
        assert len(point) == self._n_dims

        def insert_impl(data: np.ndarray, node: KDNode, cd: int):
            if isinstance(node, Leaf):
                return self.build(np.r_[node.data, data[None, :]], cd, self._n_dims)
            elif data[cd] == node.data[0, cd]:
                node.data = np.r_[node.data, data[None, :]]
            elif data[cd] < node.data[0, cd]:
                node.left = insert_impl(data, node.left, (cd + 1) % self._n_dims)
            else:
                node.right = insert_impl(data, node.right, (cd + 1) % self._n_dims)
            return node

        self._root = insert_impl(point, self._root, 0)
        self._size += 1
        self._update_region_after_insert(point)

    def access(
        self, point, default: Optional[KDNode] = Leaf(np.empty(0))
    ) -> Optional[KDNode]:
        """Find the Node in the KDTree with the given point and None if it doesn't exist."""
        assert len(point) == self._n_dims

        def access_impl(node, dim):
            nonlocal point

            if np.any(node.data == point):
                return node
            if isinstance(node, InternalNode):
                if point[dim] < node.data[0, dim]:
                    return access_impl(node.left, (dim + 1) % self._n_dims)
                else:
                    return access_impl(node.right, (dim + 1) % self._n_dims)
            return default

        return access_impl(self._root, 0)

    def __contains__(self, item):
        return self.access(item, None) is not None

    def _update_region_after_insert(self, point: np.ndarray):
        self._region = Orthotope(
            [
                Interval(min(interval.start, point_axis), max(interval.end, point_axis))
                for point_axis, interval in zip(point, self._region.intervals)
            ]
        )  # O(k lg n)

    def minimum(self, dim):
        return self._region.intervals[dim].start

    def maximum(self, dim):
        return self._region.intervals[dim].end

    def _update_region_after_remove(self, point: np.ndarray):
        for axis, (value, interval) in enumerate(zip(point, self._region.intervals)):
            if value == self.maximum(axis):
                self._region.intervals[axis] = Interval(
                    interval.start, self.max_value(axis)
                )
            elif value == self.minimum(axis):
                self._region.intervals[axis] = Interval(
                    self.min_value(axis), interval.end
                )

    def nearest_neighbor(self, point: np.ndarray) -> NNResult:
        """Find the single nearest neighbor to the point given or None if the tree is empty."""
        assert len(point) == self._n_dims

        best = NNResult(np.empty(0), np.infty)

        def search(*, node: KDNode, depth: int):
            nonlocal best, point

            if node.data.size > 0:
                local_best = min(
                    node.data, key=lambda value: self._dist_fn(value, point)
                )
                if (distance := self._dist_fn(local_best, point)) < best.distance:
                    best = NNResult(local_best, distance)

            if isinstance(node, InternalNode):
                axis = depth % self._n_dims

                if point[axis] <= node.data[0, axis]:
                    close, away = node.left, node.right
                else:
                    close, away = node.right, node.left

                search(node=close, depth=depth + 1)

                if self._dist_fn(node.data[0, axis], point[axis]) < best.distance:
                    search(node=away, depth=depth + 1)

        search(node=self._root, depth=0)
        return best

    def k_nearest_neighbors(self, point: np.ndarray, k: int) -> list[NNResult]:
        """Find K nearest neighbors of a node."""
        assert k > 0

        queue = BoundedPriorityQueue(k, point, self._dist_fn)

        def search(*, node: KDNode, depth: int):
            nonlocal queue

            queue.extend(node.data)

            if isinstance(node, InternalNode):
                axis = depth % self._n_dims

                if point[axis] <= node.data[0, axis]:
                    close, away = node.left, node.right
                else:
                    close, away = node.right, node.left

                search(node=close, depth=depth + 1)

                if (
                    not queue.is_full()
                    or self._dist_fn(node.data[0, axis], point[axis])
                    < queue.peek().distance
                ):
                    search(node=away, depth=depth + 1)

        search(node=self._root, depth=0)
        # return [NNResult(item, dist) for dist, item in queue]
        return queue

    def min_node(self, dim: int):
        """find the point with the smallest value in the dth dimension.
        This method assumes the data is not sorted"""

        assert dim < self._n_dims
        return self._root.min(dim, 0, self._n_dims)

    def max_node(self, dim: int):
        """find the point with the largest value in the dth dimension.
        This method assumes the data is not sorted"""

        assert dim < self._n_dims
        return self._root.max(dim, 0, self._n_dims)

    def min_value(self, dim: int):
        return self.min_node(dim)[dim]

    def max_value(self, dim: int):
        return self.max_node(dim)[dim]

    def remove(self, point):
        assert len(point) == self._n_dims

        def remove_impl(node: KDNode, query_point: np.ndarray, cd: int) -> KDNode:
            if isinstance(node, Leaf):
                for index, value in enumerate(node.data):
                    if np.array_equal(value, query_point):
                        node.data = np.r_[node.data[:index], node.data[index + 1 :]]
                        return node
                if isinstance(node, Leaf):
                    raise ValueError(f"point {query_point} not found")
            else:
                next_cd = (cd + 1) % self._n_dims

                for index, value in enumerate(node.data):
                    if np.array_equal(value, query_point):
                        node.data = np.r_[node.data[:index], node.data[index + 1 :]]
                        # replace this node with its successor in the same dimension
                        if node.data.size == 0:
                            # use min(cd) from right subtree
                            if node.right:
                                data = node.right.min(cd, next_cd, self._n_dims)
                                node.data = data[None, :]
                                node.right = remove_impl(node.right, data, next_cd)
                            elif node.left:
                                # swap subtrees and use min(cd) from new right
                                node.right, node.left = node.left, node.right
                                data = node.right.min(cd, next_cd, self._n_dims)
                                node.data = data[None, :]
                                node.right = remove_impl(node.right, data, next_cd)
                        break
                else:
                    if query_point[cd] < node.data[0, cd]:
                        node.left = remove_impl(node.left, query_point, next_cd)
                    else:
                        node.right = remove_impl(node.right, query_point, next_cd)

                if not node.left and not node.right:
                    return Leaf(node.data)
                else:
                    return node

        self._root = remove_impl(self._root, point, 0)
        self._size -= 1
        self._update_region_after_remove(point)

    def range_search(self, query: Orthotope):
        """Returns all the nodes in the given range.
        The perks of KD-Trees set in when the subtrees are augmented."""
        yield from self._root.range_search(query, self._region.copy(), 0, self._n_dims)

    def pretty_str(self):
        return "".join(self._root.yield_line("", "R"))

    def __len__(self):
        return self._size

    def __str__(self):
        return f"{self.__class__.__name__}<d={self._n_dims} dist_func={self._dist_fn.__name__}>({self._root}"


if __name__ == "__main__":
    import doctest

    doctest.testmod()

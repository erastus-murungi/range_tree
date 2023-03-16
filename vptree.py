from abc import ABC, abstractmethod
from collections import deque
from random import sample

import numpy as np

from kdtree import BoundedPriorityQueue, l2_norm


class VantagePointRule(ABC):
    @abstractmethod
    def get(self, points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        pass


class RandomVantagePoint(VantagePointRule):
    def get(self, points: np.ndarray):
        random_index = np.random.randint(0, len(points))
        return points[random_index], np.delete(points, random_index, axis=0)


class BestSpreadVantagePoint(VantagePointRule):
    def get(self, points):
        # traditional vantage point selection method
        # picks the point with the largest spread
        num_samples = num_tests = max(10, len(points) // 1000)

        list_points = list(points)
        sampled_points = sample(list_points, num_samples)

        best_spread = 0
        best_point_index = None
        best_point = None
        for index, point in enumerate(sampled_points):
            rand_points = sample(list_points, num_tests)
            distances = np.linalg.norm(point - rand_points, axis=1)
            mu = np.median(distances)

            # can also use np.var. I think
            spread = np.std(distances - mu)
            if spread > best_spread:
                best_spread = spread
                best_point_index = index
                best_point = point

        return best_point, np.delete(points, best_point_index, axis=0)


class VPTree:
    def __init__(
        self,
        points: np.ndarray,
        leaf_size: int = 16,
        vantage_point_rule: VantagePointRule = RandomVantagePoint(),
    ):
        if not len(points):
            raise ValueError("we cannot create a node from no points")

        self.radius = None
        self.right = None
        self.left = None
        self.vantage_point, points = vantage_point_rule.get(points)

        #  build vantage point tree
        if len(points) < 1:
            return
        if len(points) <= leaf_size:
            self.children = points
            return

        # get distances from vantage point to every point
        distances = np.linalg.norm(points - self.vantage_point, axis=1)
        self.radius = np.median(distances)

        inside, outside = (
            points[distances < self.radius],
            points[distances >= self.radius],
        )

        # can be run in parallel
        if len(inside) > 0:
            self.left = VPTree(inside, leaf_size, vantage_point_rule)
        if len(outside) > 0:
            self.right = VPTree(outside, leaf_size, vantage_point_rule)

    def k_nearest_neighbors(self, query_point, k: int = 1):

        region_radius = np.inf
        explore_queue = deque([self])
        seen_count = 0
        results_queue = BoundedPriorityQueue(k, query_point, l2_norm)

        while explore_queue:
            if (current_node := explore_queue.popleft()) is None:
                continue

            dist = l2_norm(current_node.vantage_point, query_point)
            seen_count += 1

            if dist < region_radius:
                results_queue.append(current_node.vantage_point, dist)
            if results_queue.is_full():
                region_radius = results_queue.peek().distance
            if current_node.is_leaf():
                seen_count += len(current_node.children)
                results_queue.extend(current_node.children)
                if results_queue.is_full():
                    region_radius = results_queue.peek().distance
                continue

            if dist < current_node.radius:
                explore_queue.append(current_node.left)
                if current_node.radius - dist <= region_radius:
                    explore_queue.append(current_node.right)
            else:
                explore_queue.append(current_node.right)
                if dist - current_node.radius <= region_radius:
                    explore_queue.append(current_node.left)
        return results_queue, seen_count

    def is_leaf(self):
        return self.right is None and self.left is None

    def __repr__(self):
        return f"({self.vantage_point}, {self.left.vantage_point}, {self.right.vantage_point})"


if __name__ == "__main__":
    import doctest

    doctest.testmod()

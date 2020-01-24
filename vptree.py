import numpy as np
from random import sample
from scipy.spatial.distance import minkowski
from bpq import BQPList
from heapq import heappush, heappop


class VPTree:
    def __init__(self, points, distance_function, leafsize=16):
        self.mu = None
        self.right = None
        self.left = None
        self.vantage_point = self.select_vantage_point(points, distance_function)

        #  build vantage point tree

        if len(points) < 1:
            return
        if len(points) <= leafsize:
            self.children = points
            return

        # get distances from vantage point to every point
        distances = [distance_function(self.vantage_point, p) for p in points]
        self.mu = np.median(distances)

        inside, outside = [], []
        for i, point in enumerate(points):
            if distances[i] < self.mu:
                inside.append(point)
            else:
                outside.append(point)

        # try parallelizing this step
        if len(inside) > 0:
            self.left = VPTree(inside, distance_function)
        if len(outside) > 0:
            self.right = VPTree(outside, distance_function)

    @staticmethod
    def addpoint(stack, node, query_point, distance=minkowski):
        if node is not None:
            dist = distance(node.vantage_point, query_point)
            heappush(stack, (dist, node))

    @staticmethod
    def select_vantage_point(points, distance_function):
        if len(points) <= 10:
            return points.pop(np.random.randint(0, len(points)))
        else:
            # traditional vantage point selection method
            # picks the point with the largest spread
            num_samples = num_tests = max(10, len(points) // 1000)
            sampled_points = sample(points, num_samples)

            best_spread = 0
            best_point = None
            for point in sampled_points:
                rand_points = sample(points, num_tests)
                distances = [distance_function(point, rand_point) for rand_point in rand_points]
                mu = np.median(distances)

                # can also use np.var. I think
                spread = np.std(distances - mu)
                if spread > best_spread:
                    best_spread = spread
                    best_point = point

            points.remove(best_point)
            return best_point

    def k_nearest_neighbors(self, query, k=1, distance=minkowski):
        queue = BQPList(k, query)
        tau = np.inf
        tovisit = []
        self.addpoint(tovisit, self, query)

        seen = 0

        # faster than stack
        while tovisit:
            dist, node = heappop(tovisit)
            seen += 1
            if node is None:
                continue

            if dist < tau:
                queue.push(node.vantage_point, dist)
            if queue.isfull:
                tau, _ = queue.peek()
            if node.isleaf:
                seen += len(node.children)
                for child in node.children:
                    d = distance(child, query)
                    queue.push(child, d)
                if queue.isfull:
                    tau, _ = queue.peek()
                continue

            if dist < node.mu:
                self.addpoint(tovisit, node.left, query)
                if node.mu - dist <= tau:
                    self.addpoint(tovisit, node.right, query)
            else:
                self.addpoint(tovisit, node.right, query)
                if dist - node.mu <= tau:
                    self.addpoint(tovisit, node.left, query)
        return queue.iteritems(), seen

    @property
    def isleaf(self):
        return self.right is None and self.left is None

    def height(self):
        if self.isleaf:
            return 0
        left = self.left.height() if self.left is not None else 0
        right = self.right.height() if self.right is not None else 0
        return max(left, right) + 1

    def s_value(self):
        if self.isleaf:
            return 0
        left = self.left.s_value() if self.left is not None else 0
        right = self.right.s_value() if self.right is not None else 0
        return min(left, right) + 1

    def size(self):
        if self.isleaf:
            return len(self.children) + 1
        left = self.left.size() if self.left is not None else 0
        right = self.right.size() if self.right is not None else 0
        return left + right + 1

    def __repr__(self):
        return f"({self.vantage_point}, {self.left.vantage_point}, {self.right.vantage_point})"


if __name__ == '__main__':
    num_points = 9980
    coords = [(np.random.randint(0, 100000), np.random.randint(0, 100000)) for _ in range(num_points)]
    vp = VPTree(coords, minkowski)
    q_point = (500, 1000)
    print(vp.size())
    x, total_seen = vp.k_nearest_neighbors(q_point, 4)
    print(list(x), total_seen)

    # print(repr(vp))

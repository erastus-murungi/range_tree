from operator import sub, itemgetter
from dataclasses import dataclass
from typing import Union
from sys import maxsize
from itertools import chain
from heapq import *
import numpy as np


__author__ = "Erastus Murungi"
__email__ = "murungi@mit.edu"

LEFT, RIGHT = 0, 1
LOW, HIGH = 0, 1


@dataclass
class KDNode:
    data: tuple
    left: Union[None, "KDNode"]
    right: Union[None, "KDNode"]

    # the node can be augmented with satellite information here
    # a parent pointer can be added here to enable upward traversal

    @property
    def isleaf(self):
        return self.left is None and self.right is None

    def __iter__(self):
        yield from self.data


class KDTree:
    def __init__(self, points=()):
        """The tree can be initialized with or without points"""
        self.k = len(points[0])
        self.INF = np.full(self.k, maxsize)
        self.NEG_INF = np.full(self.k, -maxsize)
        self.root = self.build_kd_tree(points, 0)

        # calculate the size of the region
        self.mins = [self.find_min(axis)[axis] for axis in range(self.k)]  # O(k lg n)
        self.maxs = [self.find_max(axis)[axis] for axis in range(self.k)]  # O(k lg n)
        # stores the total size of the tree
        self.size = len(points)

    def build_kd_tree(self, points, depth):
        # Order O(n^2 lg n) build time
        # Can be sped up using a linear time median_low finding algorithm
        # Or maintaining a k sorted lists of points to enable quick median finding. The construction of time f such
        # an algorithm is O(k nlg n), where k is the dimension

        if not points:
            return None
        axis = depth % self.k

        points.sort(key=itemgetter(axis))
        mid = len(points) >> 1

        # to ensure all duplicate_points larger than the median go to the right
        while mid > 0 and (points[mid][axis] == points[mid - 1][axis]):
            mid -= 1

        return KDNode(points[mid], self.build_kd_tree(points[:mid], depth + 1),
                      self.build_kd_tree(points[mid + 1:], depth + 1))

    def insert(self, point):
        """Inserts a single point into the KD-Tree. It does not allow duplicates"""
        assert len(point) == self.k

        def __insert(data, node, cd):
            if node is None:
                node = KDNode(data, None, None)
            elif data == node.data:
                raise ValueError("Duplicate point")
            elif data[cd] < node.data[cd]:
                node.left = __insert(data, node.left, (cd + 1) % self.k)
            else:
                node.right = __insert(data, node.right, (cd + 1) % self.k)
            return node

        res = __insert(point, self.root, 0)
        self.size += 1
        self.__insert_update_bounds(point)
        return res

    def access(self, point):
        """Find the Node in the KDTree withe the given point and None if it doesn't exist."""
        assert len(point) == self.k
        return self.__access(point, self.root, 0)

    def __access(self, point, node, dim):
        if node is None:
            return None
        elif node.data == point:
            return node
        elif point[dim] < node.data[dim]:
            return self.__access(point, node.left, (dim + 1) % self.k)
        else:
            return self.__access(point, node.right, (dim + 1) % self.k)

    def __contains__(self, item):
        return self.access(item) is not None

    def __insert_update_bounds(self, point):
        for axis in range(self.k):
            self.mins[axis] = min(self.mins[axis], point[axis])
            self.maxs[axis] = max(self.maxs[axis], point[axis])

    def minimum(self, dim):
        # assert dim < self.k
        return self.mins[dim]

    def maximum(self, dim):
        """This is faster than find_max"""
        return self.maxs[dim]

    def __delete_update_bounds(self, point):
        """Update the mins and the maxes after deleting a point from the KD-Tree"""
        for axis in point:
            if point[axis] == self.maximum(axis):
                self.maxs[axis] = self.find_max(axis)[axis]
            else:
                if point[axis] == self.minimum(axis):
                    self.mins[axis] = self.find_min(axis)[axis]

    def __get_closer_point(self, pivot, p1, p2):
        """Returns the point which is closer to the pivot"""
        if p1 is None:
            return p2
        if p2 is None:
            return p1
        return p1 if self.minkowski_distance(pivot, p1) < self.minkowski_distance(pivot, p2) else p2

    def nearest_neighbor(self, point):
        """Find the single nearest neighbor to the point given or None if the tree is empty."""
        assert len(point) == self.k
        return self.__nearest_neighbor(self.root, point)

    def __nearest_neighbor(self, node, point, cd=0):
        if node is None:
            return None

        axis = cd % self.k
        prefer, alternate = (node.left, node.right) if point[axis] < node.data[axis] else (node.right, node.left)
        best = self.__get_closer_point(point, node.data,
                                       self.__nearest_neighbor(prefer, point, cd + 1))

        # we check whether the candidate hypersphere based on
        # our current guess could cross the splitting hyperplane of the current node
        radius = abs(node.data[axis] - point[axis])
        if self.minkowski_distance(point, best) > radius:
            best = self.__get_closer_point(point, best,
                                           self.__nearest_neighbor(alternate, point, cd + 1))
        return best

    def k_nearest_neighbors(self, point):
        """Find K nearest neighbors of a node."""
        pass

    def find_min(self, dim):
        """find the point with the smallest value in the dth dimension.
        This method assumes the data is not sorted"""
        assert dim < self.k
        return self.__find_min(self.root, dim, 0)

    def __find_min(self, node, dim, axis):
        if node is None:
            return self.INF
        # T splits on the dimension we’re searching
        # => only visit left subtree
        if dim == axis:
            if node.left is None:
                return node.data
            else:
                return self.__find_min(node.left, dim, (axis + 1) % self.k)
        # T splits on a different dimension
        # => have to search both subtrees
        else:
            left_min = self.__find_min(node.left, dim, (axis + 1) % self.k)
            right_min = self.__find_min(node.right, dim, (axis + 1) % self.k)
            local = node.data
            min_ = min(left_min, right_min, local, key=lambda p: p[dim])
            return min_

    def find_max(self, dim):
        """Finds the maximum point in the tree."""
        assert dim < self.k
        return self.__find_max(self.root, dim, 0)

    def __find_max(self, node, dim, axis):
        if node is None:
            return self.NEG_INF
        if dim == axis:
            if node.right is None:
                return node.data
            else:
                return self.__find_max(node.right, dim, (axis + 1) % self.k)
        else:
            right_max = self.__find_max(node.right, dim, (axis + 1) % self.k)
            left_max = self.__find_max(node.left, dim, (axis + 1) % self.k)
            local = node.data
            max_ = max(right_max, left_max, local, key=lambda p: p[dim])
            return max_

    def remove(self, point):
        assert len(point) == self.k
        self.size -= 1
        self.__delete_update_bounds(point)

    def __remove(self, point, node, cd=0):
        if node is None:
            # we can't return anything, because doing so will modify the Tree
            raise ValueError("Node not found")

        next_cd = (cd + 1) % self.k

        # we are here
        if point == node.data:
            # replace this node with its successor in the same dimension
            if node.right is not None:
                node.data = self.__find_min(node.right, dim=cd, axis=next_cd)
                node.right = self.__remove(node.data, node.right, next_cd)
            # swap subtrees,
            if node.left is not None:
                node.data = self.__find_min(node.left, dim=cd, axis=next_cd)
                # remember that if this 'if statement' is called, then node.right must have been None
                node.right = self.__remove(node.data, node.left, next_cd)
            # since both node.right and node.left are None, we are at a leaf node
            else:
                node = None

        # we are not there yet
        # this step is just like the insertion traversal
        elif point[cd] < node.data[cd]:
            node.left = self.__remove(point, node.left, next_cd)
        else:
            node.right = self.__remove(point, node.right, next_cd)
        return node

    @staticmethod
    def __get_regions(region, split_node, axis):
        """Uses exactly one extra array on each level. We reuse the old array. The size of each array = k*2.
        The number of copies are roughly n/2."""
        region_right = np.copy(region)

        # create alias
        region_left = region
        region_left[axis][HIGH] = min(region[axis][HIGH], split_node.data[axis])
        region_right[axis][LOW] = region[axis][HIGH] + 1

        return region_left, region_right

    @staticmethod
    def node_is_contained(node, query):
        """Returns true if a node is in a certain region."""
        for axis in range(len(query)):
            if not (query[axis][LOW] <= node.data[axis] < query[axis][HIGH]):
                return False
        return True

    @staticmethod
    def __dim_is_contained(query, region, axis):
        """Checks whether one dimension in the target region is contained inside the query dimension."""
        return query[axis][LOW] <= region[axis][LOW] and query[axis][HIGH] > region[axis][HIGH]

    def region_is_contained(self, query, region):
        """Checks whether the target region fully contained inside the query hyper-rectangle"""
        return all([self.__dim_is_contained(query, region, axis) for axis in range(len(query))])

    def regions_intersect(self, query, region):
        """Checks whether there exists any intersection between two regions"""
        return any([not self.__dim_is_contained(query, region, axis) for axis in range(len(query))])

    def recalc_size(self):
        """Manually recalculates the size of the KD Tree. It is used only to check that __len__ works as expected."""
        def size_helper(node):
            if node is None:
                return 0
            else:
                return 1 + size_helper(node.left) + size_helper(node.right)

        return size_helper(self.root)

    def range_search(self, queries):
        """Returns all the nodes in the given range. The perks of KD-Trees set in when the subtrees are augmented."""
        if len(queries) != self.k:
            raise ValueError("Invalid query dimensions.")
        queries = np.array(queries)
        region = np.array(list(zip(self.mins, self.maxs)))
        result = []
        self.__range_search(self.root, region, queries, 0, result)
        return result

    def __range_search(self, v, rv, query, axis, output):
        if v is not None:
            if self.node_is_contained(v, query):
                output.append(v.data)
            if not v.isleaf:
                region_left, region_right = self.__get_regions(rv, v, axis)
                if self.region_is_contained(query, region_left):
                    output.append(v.left)
                else:
                    if self.regions_intersect(query, region_left):
                        self.__range_search(v.left, region_left, query, (axis + 1) % self.k, output)

                if self.region_is_contained(query, region_right):
                    output.append(v.right)
                else:
                    if self.regions_intersect(query, region_right):
                        self.__range_search(v.right, region_right, query, (axis + 1) % self.k, output)

    def __str__(self):
        if self.root is None:
            return str(None)
        else:
            self.__print_helper(self.root, "", True)
            return ''

    def __print_helper(self, node, indent, last):
        """Simple recursive tree printer"""
        if node is None:
            print(indent + "∅")
        else:
            if node.isleaf:
                print(indent, end='')
                if last:
                    print("R----", end='')
                else:
                    print("L----", end='')
                print(str(node.data))
            else:
                print(indent, end='')
                if last:
                    print("R----", end='')
                    indent += "     "
                else:
                    print("L----", end='')
                    indent += "|    "
                print(str(node.data))
                self.__print_helper(node.left, indent, False)
                self.__print_helper(node.right, indent, True)

    def __len__(self):
        return self.size

    def repr(self):
        return repr(self.root)

    @staticmethod
    def minkowski_distance(points1, points2, p=2):
        if any([p is None or p >= maxsize for p in chain(points1, points2)]):
            raise ValueError("Invalid points")
        return (sum([abs(sub(*z)) ** p for z in zip(points1, points2)])) ** (1 / p)

    @staticmethod
    def report(output: list):
        """Generates the nodes in the subtrees in the order in which they were found."""

        def __report_helper(n):
            if n is not None:
                if n.left is not None:
                    yield from __report_helper(n.left)
                yield n.data
                if n.right is not None:
                    yield from __report_helper(n.right)

        for node in output:
            if type(node) is tuple:
                yield node
            else:
                yield from __report_helper(node)


def brute_nearest_neighbor(coords, p, distance_function):
    # naive nearest neighbor
    best_dist, best_point = maxsize, None
    for coord in coords:
        dist = distance_function(coord, p)
        if dist < best_dist:
            best_dist, best_point = dist, coord
    return best_point


def brute_range_search(points, query):
    for point in points:
        for axis in range(n := len(query)):
            if not (query[axis][LOW] <= point[axis] < query[axis][HIGH]):
                break
            if axis + 1 == n:
                yield point


if __name__ == '__main__':
    from random import randint
    from pprint import PrettyPrinter
    from datetime import datetime

    pp = PrettyPrinter(indent=5)
    from pympler import asizeof

    d = 5
    lim = 10000
    num_coords = 100000
    test_rounds = 1


    def randy():
        return randint(0, lim)


    for _ in range(test_rounds):
        coordinates = [tuple([randy() for _ in range(d)]) for _ in range(num_coords)]
        # coordinates = [(1, 60, 65), (9, 63, 25),
        #                (14, 6, 64), (43, 38, 32), (44, 43, 87), (49, 39, 63), (66, 24, 14), (74, 35, 3), (76, 99, 22),
        #                (77, 24, 10)]

        # print(sorted(coordinates))
        qs = ((0, 5000), (0, 1000), (0, 5000), (1000, 2000), (700, 80000))
        kdtree = KDTree(coordinates)
        output = kdtree.range_search(qs)
        brute = list(brute_range_search(coordinates, qs))
        print(len(brute))
        print(len(list(kdtree.report(output))))
        # print(kdtree)

        print("The object uses:", f"{asizeof.asizeof(kdtree) / (2 ** 20):.2f} MB for {num_coords}"
                                  f" {d}-D points.")
        # # print(kdtree)
        # p = (90, 78, 200, 409, 499)
        # t1 = datetime.now()
        # nn = kdtree.nearest_neighbor(p)
        # print(f"kd-tree NN query ran in {(datetime.now() - t1).total_seconds()}.")
        #
        # t2 = datetime.now()
        # bnn = brute_nearest_neighbor(coordinates, p, KDTree.minkowski_distance)
        # print(f"brute NN query ran in {(datetime.now() - t2).total_seconds()}.")
        #
        # assert bnn == nn
        #
        # for coord in coordinates:
        #     assert (coord in kdtree)
        # # print(y.data)
        # # x = kdtree.find_min(dim=0)
        # # print(kdtree.recalc_size())
        # # print(len(kdtree))
        # # pp.pprint(repr(kdtree.root))

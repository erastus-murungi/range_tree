from operator import sub, itemgetter
from dataclasses import dataclass
from typing import Union


__author__ = "Erastus Murungi"
__email__ = "murungi@mit.edu"


@dataclass
class KDNode:
    data: tuple
    left: Union[None, "KDNode"]
    right: Union[None, "KDNode"]


class KDTree:
    INF = float('inf')

    def __init__(self, points=()):
        self.k = len(points[0])
        self.root = self.build_kd_tree(points, 0)
        self.size = len(points)

    def build_kd_tree(self, points, depth):
        if not points:
            return None
        axis = depth % self.k

        points.sort(key=itemgetter(axis))
        mid = len(points) >> 1

        return KDNode(points[mid], self.build_kd_tree(points[:mid], depth + 1),
                      self.build_kd_tree(points[mid + 1:], depth + 1))

    def insert(self, point):
        assert len(point) == self.k
        self.size += 1

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
        return res

    def find(self, point):
        assert len(point) == self.k
        return self.__find(point, self.root, 0)

    def __find(self, point, node, dim):
        if node is None:
            return None
        elif node.data == point:
            return node
        elif point[dim] < node.data[dim]:
            return self.__find(point, node.left, (dim + 1) % self.k)
        else:
            return self.__find(point, node.right, (dim + 1) % self.k)

    def __contains__(self, item):
        return self.find(item) is not None

    def nearest_neighbor(self):
        pass

    def find_min(self, dim):
        """find the point with the smallest value in the dth dimension.
        This method assumes the data is not sorted"""
        return self.__find_min(self.root, dim, 0)

    def __find_min(self, node, dim, axis):
        if node is None:
            return None
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
            return min(self.__find_min(node.left, dim, (axis + 1) % self.k),
                       self.__find_min(node.right, dim, (axis + 1) % self.k),
                       node.data, key=lambda t: self.INF if t is None else t[dim])

    def remove(self, point):
        assert len(point) == self.k
        something = self.__remove(point, self.root)
        self.size -= 1
        return something

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

    def recalc_size(self):
        def size_helper(node):
            if node is None:
                return 0
            else:
                return 1 + size_helper(node.left) + size_helper(node.right)
        return size_helper(self.root)
    
    def range_search(self, queries):
        pass

    def __len__(self):
        return self.size

    def repr(self):
        return repr(self.root)

    @staticmethod
    def minkowski_distance(points1, points2, p=2):
        return (sum([abs(sub(*z)) ** p for z in zip(points1, points2)])) ** (1 / p)


if __name__ == '__main__':
    from random import randint
    from pprint import PrettyPrinter
    pp = PrettyPrinter(indent=5)

    lim = 100
    num_coords = 10


    def randy():
        return randint(0, lim)

    # coordinates = [(randy(), randy()) for _ in range(num_coords)]
    coordinates = [(3, 40), (4, 83), (14, 14), (37, 51), (43, 48), (59, 86), (61, 82), (69, 93), (93, 9), (99, 34)]
    print(sorted(coordinates))
    kdtree = KDTree(coordinates)
    kdtree.insert((1, 30))
    for coord in coordinates:
        print(coord in kdtree)
    print((1, 30) in kdtree)
    y = kdtree.find((1, 30))
    print(y.data)
    kdtree.remove((1, 30))
    x = kdtree.find_min(dim=0)
    print(x)
    print(kdtree.recalc_size())
    print(len(kdtree))
    pp.pprint(repr(kdtree.root))

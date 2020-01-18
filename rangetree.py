from abc import *


def brute_algorithm(coords, x1, x2, y1, y2):
    for x, y in coords:
        if x1 <= x < x2 and y1 <= y < y2:
            yield x, y


class RangeTree(ABC):
    def __init__(self):
        self.root = None

    @staticmethod
    @abstractmethod
    def isleaf(node):
        pass

    def split_value(self, node, get) -> float:
        """This is just the maximum value in the left subtree"""

        if node is None:
            return 0
        elif self.isleaf(node):
            return get(node.point)
        else:
            return max(node.point, self.split_value(node.left, get), self.split_value(node.right, get))

    def __str__(self):
        if self.root is None:
            return str(None)
        else:
            self.__print_helper(self.root, "", True)
            return ''

    def __print_helper(self, node, indent, last):
        """Simple recursive tree printer"""
        if node is None:
            print(indent)
            print(None)
        else:
            if self.isleaf(node):
                print(indent, end='')
                if last:
                    print("R----", end='')
                else:
                    print("L----", end='')
                print(str(node.point))
            else:
                print(indent, end='')
                if last:
                    print("R----", end='')
                    indent += "     "
                else:
                    print("L----", end='')
                    indent += "|    "
                print(str(node.point))
                self.__print_helper(node.left, indent, False)
                self.__print_helper(node.right, indent, True)

    def report(self, output: list):
        """Generates the nodes in the subtrees in the order in which they were found."""

        def __report_helper(n):
            if n is not None:
                if self.isleaf(n.left):
                    yield n.left.point
                else:
                    yield from __report_helper(n.left)
                if self.isleaf(n.right):
                    yield n.right.point
                else:
                    yield from __report_helper(n.right)

        for node in output:
            if type(node) == tuple:
                yield node
            else:
                if self.isleaf(node):
                    yield node.point
                else:
                    yield from __report_helper(node)

    def find_split_node(self, x, y, getpoint=lambda y: y.point):
        """ Finds and returns the split node
            For the range query [x : x'], the node v in a balanced binary search
            tree is a split node if its value x.v satisfies x.v ≥ x and x.v < x'.
        """

        v = self.root
        while not self.isleaf(v) and (getpoint(v) >= y or getpoint(v) < x):
            v = v.left if y <= getpoint(v) else v.right
        return v

    def __repr__(self) -> str:
        return repr(self.root)

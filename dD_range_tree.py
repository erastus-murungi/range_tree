# d-D range tree implementation
from rangetree import *


class DDRangeTree(RangeTree):
    def __init__(self, points):
        super().__init__()
        self.d = len(points[0])
        for point in points:
            assert len(point) == self.d

    @staticmethod
    def isleaf(node):
        pass

    def build_d_range_tree(self, values):
        pass

    def build_dd_range_tree_helper(self, depth):
        if depth == self.d:
            pass

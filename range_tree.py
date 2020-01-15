from dataclasses import dataclass
from typing import Union


@dataclass
class Leaf:
    key: float
    parent: Union["Node", None] = None


@dataclass
class Node:
    left: Union[Leaf, "Node", None]
    right: Union[Leaf, "Node", None]
    key: float
    parent: Union["Node", None] = None


def isleaf(node):
    return type(node) == Leaf


def height(node):
    return -1 if node is None else 0 if isleaf(node) else max(height(node.left), height(node.right)) + 1


def max_left_subtree(node):
    return 0 if node is None \
        else node.key if isleaf(node) \
        else max(node.key, max_left_subtree(node.left), max_left_subtree(node.right))


def range_tree(values):
    """Build a 1D Range Tree and returns the root, and the nodes on the same level"""
    assert (len(values)), "Empty iterable."
    leaves = list(map(lambda val: Leaf(val, None), sorted(values)))
    if len(leaves) == 1:
        return Leaf(values[0], None)

    levels = [leaves]
    while (n := len(leaves)) > 1:
        nodes = []
        for i in range(1, n, 2):
            l, r = leaves[i - 1], leaves[i]
            x = Node(l, r, max_left_subtree(l), None)
            l.parent = r.parent = x
            nodes.append(x)
        if n & 1:  # if odd
            x = Node(leaves[n - 1], None, max_left_subtree(leaves[n - 1]), None)
            nodes.append(x)
            leaves[n - 1].parent = x
        leaves = nodes
        levels.append(leaves)

    return levels[-1][0], levels


def report(output: list):
    def __report_helper(node):
        if isleaf(node.left):
            yield node.left.key
        else:
            yield from node.left.key
        yield node.key
        if isleaf(node.right):
            yield node.right.key
        else:
            yield from node.right

    for node in output:
        if isleaf(node):
            yield node.key
        else:
            yield from __report_helper(node)


def find_split_node(root, x, y):
    """ Finds and returns the split node
        For the range query [x : y], the node v in a balanced binary search
        tree is a split node if its value x.v satisfies x.v ≥ x and x.v < x """

    v = root
    while not isleaf(v) and (v.key >= y or v.key < x):
        v = v.left if y <= v.key else v.right
    return v


def query_range_tree(root, i, j):
    """Querying a 1D Range Tree."""
    if i > j:
        i, j = j, i

    output = []
    v_split = find_split_node(root, i, j)
    if isleaf(v_split):
        # check if the key in v_split
        if i <= v_split.key <= j:  # inclusive version
            output.append(v_split)
    else:
        v = v_split.left
        while not isleaf(v):
            if i <= v.key:
                # report right subtree
                output.append(v.right)
                v = v.left
            else:
                v = v.right
        # v is now a leaf
        if v.key >= i:
            output.append(v)
        # now we follow right side
        v = v_split.right
        while not isleaf(v):
            if v.key < j:
                # report left subtree
                output.append(v.left)
                v = v.right
            else:
                v = v.left
    return output


if __name__ == '__main__':
    from pprint import pprint
    points = [3, 10, 19, 23, 30, 37, 59, 62, 70, 80, 100, 105]
    rtree, levels_ = range_tree(points)
    vals = [list(map(lambda x: x.key, values)) for values in levels_]
    vals.reverse()
    pprint(vals)

    rep = query_range_tree(rtree, 20, 60)
    gen = report(rep)
    try:
        while True:
            print(next(gen))
    except StopIteration:
        pass

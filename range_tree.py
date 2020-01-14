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
    leaves = list(map(lambda val: Leaf(val, None), values))
    if len(leaves) == 1:
        return Leaf(values[0], None)

    levels = [leaves]
    while (n := len(leaves)) > 1:
        nodes = []
        for i in range(1, n, 2):
            l, r = leaves[i - 1], leaves[i]
            x = Node(l, r, max_left_subtree(l), None)
            nodes.append(x)
        if n & 1:  # if odd
            x = Node(leaves[n - 1], None, max_left_subtree(leaves[n - 1]), None)
            nodes.append(x)
            leaves[n - 1].parent = x
        leaves = nodes
        levels.append(leaves)

    return levels[-1][0], levels


def access(subtree_root, key):
    assert subtree_root, "Null tree"
    while not isleaf(subtree_root):
        subtree_root = subtree_root.right if key >= subtree_root.key else subtree_root.left
    return subtree_root


def lca(root, node1, node2):
    pass


def query_range_tree(root, i, j):
    if i > j:
        i, j = j, i
    node_i, node_j = access(root, i), access(root, j)
    assert node_j != node_i
    output = []
    common = lca(root, node_i, node_j)
    # query left
    if node_i.key == i:
        output.append(node_i)
    node_i = node_i.parent
    while node_i is not common and node_i is not root:
        if node_i is node_i.parent.left:
            output.append(node_i.right)
            node_i = node_i.parent
    output.append(common)
    # query right
    if node_j.key == j:
        output.append(node_j)
    node_j = node_j.parent
    while node_j is not common and node_j is not root:
        if node_j is node_j.parent.right:
            output.append(node_j.left)
            node_j = node_j.parent
    return output


if __name__ == '__main__':
    from pprint import pprint
    points = [6, 15, 17, 21, 24, 33, 42, 51, 52, 57, 65, 73, 78]
    rtree, levs = range_tree(points)
    vals = [list(map(lambda x: x.key, values)) for values in levs]
    vals.reverse()
    pprint(vals)

    print(query_range_tree(rtree, 15, 51))

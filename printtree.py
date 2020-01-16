def printtree(tree):
    if tree.root is None:
        return str(None)
    else:
        __print_helper(tree, tree.root, "", True)
        return ''


def __print_helper(tree, node, indent, last):
    """Simple recursive tree printer"""
    if node is None:
        print(indent)
        print(None)
    else:
        if tree.isleaf(node):
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
            __print_helper(tree, node.left, indent, False)
            __print_helper(tree, node.right, indent, True)

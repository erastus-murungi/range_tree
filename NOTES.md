```python
def query(node, q, c):
    if node.is_leaf:
        if q.contains(node.point):
            yield node.point

    elif q.contains(c):
        yield from node.report_leaves()

    elif not q.is_disjoint_from(c):
        yield from node.less.query_axis_recursive(q, Interval(c.start, node.split_value))
        yield from node.greater.query_axis_recursive(q, Interval(node.split_value, c.end))
```
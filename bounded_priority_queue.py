from rbt import RedBlackTree


class BoundedPriorityQueue:
    """Simple bounded priority queue which uses a Red Black Tree as the underlying data structure."""

    def __init__(self, k):
        self.maxkey = -float("inf")
        if k < 0:
            raise ValueError("k should be larger than 0")
        self.k = k
        self._bpq = RedBlackTree()

    def insert(self, key, item):
        if self.k == 0:
            return self.maxkey
        if len(self._bpq) < self.k or key < self.maxkey:
            self._bpq.insert(key, item)  # O (lg n)
        if len(self._bpq) > self.k:
            self._bpq.delete(self._bpq.maximum[0])  # O(lg n)
        self.maxkey = self._bpq.maximum[0]  # (lg n)

    def __setitem__(self, key, item=0):
        assert (type(key)) in [int, float], "Invalid type of key."
        self.insert(key, item)

    @property
    def isfull(self):
        return len(self._bpq) == self.k

    def iteritems(self):
        return self._bpq.iteritems()

    def __repr__(self):
        return repr(self._bpq.root)

    def __str__(self):
        return str(list(self._bpq.iteritems()))


if __name__ == "__main__":
    from random import randint

    values = [(randint(0, 100), randint(1000, 10000)) for _ in range(10)]
    bp = BoundedPriorityQueue(1)
    print(values)
    for key, val in values:
        bp[key] = val
        print(bp)

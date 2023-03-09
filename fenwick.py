import operator as op

import numpy as np


def lsb(x):
    return x & -x


class FenwickTree:
    """I-D Fenwick tree."""

    __slots__ = ("array", "aggregator")

    def __init__(self, arr, aggregator=op.add):
        self.array = np.insert(arr, 0, 0)
        self.aggregator = aggregator
        self.construct()

    def construct(self):
        # construct
        n = len(self.array)

        for i in range(1, n):
            j = i + lsb(i)
            if j < n:
                self.array[j] = self.aggregator(self.array[i], self.array[j])

    def update(self, i, new_val):
        diff = self.array[i] - new_val
        self.array[i] = new_val
        j = i + lsb(i)
        while j < len(self.array):
            self.array[j] = self.aggregator(self.array[j], diff)
            j = i + lsb(i)

    def point_query(self, i):
        i += 1  # because of 1-based indexing
        total = 0
        while i:
            total = self.aggregator(total, self.array[i])
            i &= ~lsb(i)  # i -= lsb(i)
        return total

    def __str__(self):
        return repr(self.array)

    def range_query(self, start, end):
        return self.point_query(end) - self.point_query(start - 1)


if __name__ == "__main__":
    from random import randint
    from time import perf_counter

    nums = np.random.choice(100_000_000, 1_000_000)
    start = perf_counter()
    f_tree = FenwickTree(nums, max)
    print(f"Fenwick tree constructed in {perf_counter() - start} seconds.")

    start = perf_counter()
    y = f_tree.point_query(900_000)
    print(f"Fenwick tree queried in {perf_counter() - start} seconds.")

    start = perf_counter()
    z = np.max(nums[:900_001])
    print(f"Numpy sum done in {perf_counter() - start} seconds.")

    print(y, z)

import numpy as np


def lsb(x):
    return x & -x


class FenwickTree:
    """I-D Fenwick tree."""
    def __init__(self, arr):
        self.array = np.insert(arr, 0, 0)
        self.construct()

    def construct(self):
        # construct
        n = len(self.array)

        for i in range(1, n):
            j = i + lsb(i)
            if j < n:
                self.array[j] += self.array[i]

    def update(self, i, new_val):
        # ass
        diff = self.array[i] - new_val
        self.array[i] = new_val
        j = i + lsb(i)
        while j < len(self.array):
            self.array[j] += diff
            j = i + lsb(i)

    def point_query(self, i):
        i += 1  # because of 1-based indexing
        total = 0
        while i:
            total += self.array[i]
            i &= ~lsb(i)  # i -= lsb(i)
        return total

    def __str__(self):
        return repr(self.array)

    def range_query(self, start, end):
        return self.point_query(end) - self.point_query(start - 1)


if __name__ == '__main__':
    from random import randint
    from time import perf_counter
    nums = np.random.choice(100_000_000, 10_000_000)
    t1 = perf_counter()
    f_tree = FenwickTree(nums)
    print(f"Fenwick tree constructed in {perf_counter() - t1} seconds.")
    t2 = perf_counter()
    y = f_tree.point_query(3_000_000)
    print(f"Fenwick tree queried in {perf_counter() - t2} seconds.")
    t3 = perf_counter()
    z = np.sum(nums[:3_000_001])
    print(f"Numpy sum done in {perf_counter() - t3} seconds.")

    print(y, z)

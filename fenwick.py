import numpy as np


def lsb(x):
    return x & -x


class FenwickTree:
    """I-D Fenwick tree"""
    def __init__(self, arr):
        self.array = np.array([0] + arr)
        self.construct()

    def construct(self):
        # construct
        n = len(self.array)

        for i in range(1, n):
            j = i + lsb(i)
            if j < n:
                self.array[j] += self.array[i]

    def update(self, i, val):
        diff = self.array[i] - val

        self.array[i] = val
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
    nums = [randint(0, 100) for _ in range(100)]
    print(nums)
    f_tree = FenwickTree(nums)
    y = f_tree.point_query(2)
    print(y

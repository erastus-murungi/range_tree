from bisect import insort

from scipy.spatial.distance import minkowski


class BPQList:
    """Fast list-based bounded priority queue."""

    def __init__(self, k, base, distance_function=minkowski):
        self.bpq = []
        self.k = k
        self._base = base
        self.distance_function = distance_function

    def push(self, coord, dist=None):
        if dist is None:
            dist = self.distance_function(coord, self._base)
        if len(self.bpq) < self.k or dist < self.bpq[-1][0]:
            insort(self.bpq, (dist, coord))
            if len(self.bpq) > self.k:
                self.bpq.pop()

    @property
    def isfull(self):
        return len(self.bpq) == self.k

    def shrink(self, new_k):
        if new_k > self.k:
            raise ValueError("New bound should be less")
        self.bpq = self.bpq[:new_k]

    def __setattr__(self, name, value):
        if name == "self._base":
            raise ValueError("{} is frozen".format(name))
        else:
            object.__setattr__(self, name, value)

    def peek(self):
        if len(self.bpq) > 0:
            return self.bpq[-1]

    def iteritems(self):
        yield from self.bpq

    def getpoints(self):
        return list(map(lambda x: x[1], self.bpq))

    def __len__(self):
        return len(self.bpq)


if __name__ == "__main__":
    from random import randint

    coords = [(randint(0, 1000), randint(0, 1000)) for _ in range(10)]
    reference_point = (100, 100)
    bpq = BPQList(3, reference_point)
    for coord in coords:
        print(bpq.isfull)
        bpq.push(coord)
    print(list(bpq.iteritems()))
    print(bpq.peek())

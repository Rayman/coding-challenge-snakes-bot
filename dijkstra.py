from collections import deque
from typing import List

import numpy as np

from .utils import is_on_grid, neighbors


def dijkstra(start, grid):
    assert is_on_grid(start, grid.shape)

    queue = deque()
    # queue = LinearQueue()
    # queue = CircularQueue()

    dist = [np.iinfo(int).max] * grid.size
    dist[start[0] * grid.shape[1] + start[1]] = 0

    queue.append(start)

    while queue:
        current = queue.popleft()
        new_dist = dist[current[0] * grid.shape[1] + current[1]] + 1
        # print(f'inspecting {current} new_dist={new_dist}')

        for neighbor in neighbors(*current, grid):
            i = neighbor[0] * grid.shape[1] + neighbor[1]
            if new_dist < dist[i]:
                dist[i] = new_dist
                # print(f'pushing {neighbor}')
                queue.append(neighbor)

    return np.array(dist).reshape(-1, grid.shape[1])


class LinearQueue:
    __slots__ = ['queue', 'first']

    def __init__(self):
        self.queue = []
        self.first = 0

    def put(self, item):
        """Put the item on the queue."""
        self.queue.append(item)

    def get(self):
        """Remove and return an item from the queue."""
        item = self.queue[self.first]
        self.first += 1
        assert self.first <= len(self.queue)
        return item

    def __len__(self):
        return len(self.queue) - self.first


class CircularQueue:
    __slots__ = ['queue', 'first', 'last', 'len']

    def __init__(self):
        self.queue = [None] * 32
        self.first = 0
        self.last = 0
        self.len = 0

    def put(self, item):
        """Put the item on the queue."""
        assert self.len < len(self.queue)
        self.queue[self.last] = item
        self.last = (self.last + 1) & 31
        self.len += 1

    def get(self):
        """Remove and return an item from the queue."""
        item = self.queue[self.first]
        self.first = (self.first + 1) & 31
        self.len -= 1
        return item

    def _increment(self, i):
        return (i + 1) & 31

    def __len__(self):
        return self.len


def print_array(grid: np.array):
    a = grid.astype(float)
    a[a == np.iinfo(int).max] = np.inf
    print(np.array_str(np.flipud(a.T), max_line_width=np.inf))


def print_list(grid: List, cols):
    a = np.array(grid).reshape((-1, cols))
    print_array(a)

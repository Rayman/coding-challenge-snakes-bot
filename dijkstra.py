from collections import deque
from typing import List

import numpy as np

from .utils import is_on_grid, neighbors


def dijkstra(start, grid):
    assert is_on_grid(start, grid.shape)

    queue = deque()

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


def print_array(grid: np.array):
    a = grid.astype(float)
    a[a == np.iinfo(int).max] = np.inf
    print(np.flipud(a.T))


def print_list(grid: List, cols):
    a = np.array(grid).reshape((-1, cols))
    print_array(a)

from collections import deque

import numpy as np

from .utils import is_on_grid, neighbors


def dijkstra(start, grid):
    assert is_on_grid(start, grid.shape)

    queue = deque()

    dist = np.full_like(grid, np.iinfo(int).max, dtype=int)
    dist[start[0], start[1]] = 0

    queue.append(start)

    while queue:
        current = queue.popleft()
        # print(f'inspecting {current}')

        new_dist = dist[current[0], current[1]] + 1
        for neighbor in neighbors(current, grid):
            if new_dist < dist[neighbor[0], neighbor[1]]:
                dist[neighbor[0], neighbor[1]] = new_dist
                # print(f'pushing {neighbor}')
                queue.append(neighbor)

    return dist

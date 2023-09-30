from collections import deque

import numpy as np

from .utils import is_on_grid
from ...constants import RIGHT, UP, LEFT, DOWN


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


def neighbors(position, grid):
    if position[0] > 0:
        yield from _possible_neighbor(position + LEFT, grid)
    if position[1] > 0:
        yield from _possible_neighbor(position + DOWN, grid)
    if position[0] < grid.shape[0] - 1:
        yield from _possible_neighbor(position + RIGHT, grid)
    if position[1] < grid.shape[1] - 1:
        yield from _possible_neighbor(position + UP, grid)


def _possible_neighbor(pos, grid):
    # print(f'collision: {self.grid[pos[0], pos[1]]}')
    if not grid[pos[0], pos[1]]:
        yield pos

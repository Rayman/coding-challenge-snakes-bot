from heapq import heappush, heappop

import numpy as np

from .utils import is_on_grid
from ...constants import RIGHT, UP, LEFT, DOWN


def shortest_path(start, goal, grid):
    """
    Calculate the shortest path from start to goal on a boolean grid

    :return: Length of the path or None
    """
    assert is_on_grid(start, grid.shape)
    assert is_on_grid(goal, grid.shape)

    if grid[goal[0], goal[1]]:
        # goal is blocked
        return None

    g_scores = np.full_like(grid, np.iinfo(int).max, dtype=int)
    g_scores[start[0], start[1]] = 0

    f_scores = np.full_like(grid, np.inf, dtype=float)
    f_scores[start[0], start[1]] = np.linalg.norm(start - goal)

    open_set = PriorityQueue()
    open_set.push(AStarNode(start, grid, f_scores))

    while open_set:
        current = open_set.pop()
        # print(f'inspecting {current}')

        if np.array_equal(current.position, goal):
            # print('goal found')
            # print(np.flipud(gScores.T))
            return g_scores[goal[0], goal[1]]

        tentative_g_score = g_scores[current.position[0], current.position[1]] + 1
        for neighbor in current.neighbors():
            if tentative_g_score < g_scores[neighbor.position[0], neighbor.position[1]]:
                # This path to neighbor is better than any previous one. Record it!
                g_scores[neighbor.position[0], neighbor.position[1]] = tentative_g_score
                f_scores[neighbor.position[0], neighbor.position[1]] = tentative_g_score + np.linalg.norm(
                    neighbor.position - goal)
                # print(f'pushing node tentative_gScore={tentative_gScore} fScore={fScores[current.position[0], current.position[1]]} gScore={gScores[current.position[0], current.position[1]]}')
                open_set.push(neighbor)

    return None


class AStarNode:
    def __init__(self, position, grid, fScores):
        self.position = position
        self.fScores = fScores
        self.grid = grid

    def neighbors(self):
        if self.position[0] > 0:
            yield from self._possible_neighbor(self.position + LEFT)
        if self.position[1] > 0:
            yield from self._possible_neighbor(self.position + DOWN)
        if self.position[0] < self.grid.shape[0] - 1:
            yield from self._possible_neighbor(self.position + RIGHT)
        if self.position[1] < self.grid.shape[1] - 1:
            yield from self._possible_neighbor(self.position + UP)

    def _possible_neighbor(self, pos):
        # print(f'collision: {self.grid[pos[0], pos[1]]}')
        if not self.grid[pos[0], pos[1]]:
            yield AStarNode(pos, self.grid, self.fScores)

    @property
    def fScore(self):
        return self.fScores[self.position[0], self.position[1]]

    def __lt__(self, other):
        return self.fScore < other.fScore

    def __str__(self):
        return f'{self.__class__.__name__}(position={self.position}, fScore={self.fScore})'


class PriorityQueue:
    __slots__ = ('heap',)

    def __init__(self):
        self.heap = []

    def push(self, item):
        heappush(self.heap, item)

    def pop(self):
        return heappop(self.heap)

    def __bool__(self):
        return bool(self.heap)

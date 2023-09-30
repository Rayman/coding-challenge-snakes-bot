import numpy as np

from .dijkstra import dijkstra


def test_dijkstra():
    start = np.array([0, 3])
    grid = np.flipud(np.array([
        [0, 0, 1, 0],
        [0, 1, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
    ], dtype=bool)).T
    dist = dijkstra(start, grid)
    assert dist[3, 3] == 7

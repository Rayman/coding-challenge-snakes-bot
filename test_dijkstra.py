import numpy as np

from .dijkstra import dijkstra, print_array


def test_dijkstra():
    start = (0, 3)
    grid = np.flipud(np.array([
        [0, 0, 1, 0],
        [0, 1, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
    ], dtype=bool)).T
    dist = dijkstra(start, grid)
    print_array(dist)
    # assert dist[3 * grid.shape[1] + 3] == 7
    assert dist[3, 3] == 7


def test_benchmark_dijkstra(benchmark):
    start = (3, 5)
    grid = np.zeros((16, 16), dtype=bool)
    dist = benchmark(dijkstra, start, grid)
    print_array(dist)

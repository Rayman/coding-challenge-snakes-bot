import numpy as np

from .shortest_path import shortest_path, dijkstra


def test_shortest_path():
    start = np.array([0, 0])
    goal = np.array([0, 1])
    grid = np.flipud(np.array([
        [0],
        [0],
    ], dtype=bool)).T
    assert 1 == shortest_path(start, goal, grid)


def test_shortest_path_blocked():
    start = np.array([0, 0])
    goal = np.array([0, 1])
    grid = np.flipud(np.array([
        [1],
        [0],
    ], dtype=bool)).T
    assert None is shortest_path(start, goal, grid)


def test_shortest_path_detour():
    start = np.array([0, 3])
    goal = np.array([3, 3])
    grid = np.flipud(np.array([
        [0, 0, 1, 0],
        [0, 1, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
    ], dtype=bool)).T
    assert 7 == shortest_path(start, goal, grid)


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

import numpy as np

from .dijkstra import dijkstra, print_array, dijkstra2


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


def calculate_voronoi_diagram(collision_grid, player_head, opponent_head):
    voronoi = dijkstra2(((player_head, 0), (opponent_head, 1)), collision_grid)
    # print_array(voronoi)
    free = ~collision_grid
    player_first = ~(voronoi & 1) & free
    opponent_first = voronoi & 1 & free
    return np.ma.masked_array(voronoi, player_first), np.ma.masked_array(voronoi, opponent_first)


def test_voronoi_heuristic():
    grid_size = (3, 3)
    """
    |x   |
    |    |
    |0  1|
    """
    collision_grid = np.zeros(grid_size, dtype=bool)
    collision_grid[0, 2] = True

    player_first, opponent_first = calculate_voronoi_diagram(collision_grid, (0, 0), (2, 0))

    print()
    print_array(player_first)
    print_array(opponent_first)

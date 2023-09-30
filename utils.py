from typing import Tuple

UP = (0, 1)
DOWN = (0, -1)
LEFT = (-1, 0)
RIGHT = (1, 0)


def is_on_grid(pos, grid_size):
    return 0 <= pos[0] < grid_size[0] and 0 <= pos[1] < grid_size[1]


def vector_add(a: Tuple, b: Tuple):
    assert len(a) == len(b) == 2
    return a[0] + b[0], a[1] + b[1]
    # return tuple(map(add, a, b))


def neighbors(position: Tuple, grid):
    assert isinstance(position, Tuple) or position.shape == (2,)
    if position[0] > 0:
        yield from _possible_neighbor(vector_add(position, LEFT), grid)
    if position[1] > 0:
        yield from _possible_neighbor(vector_add(position, DOWN), grid)
    if position[0] < grid.shape[0] - 1:
        yield from _possible_neighbor(vector_add(position, RIGHT), grid)
    if position[1] < grid.shape[1] - 1:
        yield from _possible_neighbor(vector_add(position, UP), grid)


def _possible_neighbor(pos, grid):
    assert isinstance(pos, Tuple)
    # print(f'collision: {grid[pos[0], pos[1]]}')
    if not grid[pos[0], pos[1]]:
        yield pos

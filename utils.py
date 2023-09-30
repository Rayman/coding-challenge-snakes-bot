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


def neighbors(i: int, j: int, grid):
    # assert isinstance(i, int), type(i)
    # assert isinstance(j, int), type(j)
    if i > 0:
        pos = vector_add((i, j), LEFT)
        if not grid[pos[0], pos[1]]:
            yield pos
    if j > 0:
        pos = vector_add((i, j), DOWN)
        if not grid[pos[0], pos[1]]:
            yield pos
    if i < grid.shape[0] - 1:
        pos = vector_add((i, j), RIGHT)
        if not grid[pos[0], pos[1]]:
            yield pos
    if j < grid.shape[1] - 1:
        pos = vector_add((i, j), UP)
        if not grid[pos[0], pos[1]]:
            yield pos

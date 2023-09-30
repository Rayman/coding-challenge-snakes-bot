def is_on_grid(pos, grid_size):
    return 0 <= pos[0] < grid_size[0] and 0 <= pos[1] < grid_size[1]


def neighbors(i: int, j: int, grid):
    # assert isinstance(i, int), type(i)
    # assert isinstance(j, int), type(j)
    if i > 0:
        pos = i - 1, j
        if not grid[pos[0], pos[1]]:
            yield pos
    if j > 0:
        pos = i, j - 1
        if not grid[pos[0], pos[1]]:
            yield pos
    if i < grid.shape[0] - 1:
        pos = i + 1, j
        if not grid[pos[0], pos[1]]:
            yield pos
    if j < grid.shape[1] - 1:
        pos = i, j + 1
        if not grid[pos[0], pos[1]]:
            yield pos

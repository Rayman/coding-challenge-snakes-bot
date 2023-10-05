def is_on_grid(pos, grid_size):
    return 0 <= pos[0] < grid_size[0] and 0 <= pos[1] < grid_size[1]


def neighbors(i: int, j: int, grid):
    # assert isinstance(i, int), type(i)
    # assert isinstance(j, int), type(j)
    if i > 0:
        neighbor = i - 1, j
        if not grid[neighbor]:
            yield neighbor
    if j > 0:
        neighbor = i, j - 1
        if not grid[neighbor]:
            yield neighbor
    if i < grid.shape[0] - 1:
        neighbor = i + 1, j
        if not grid[neighbor]:
            yield neighbor
    if j < grid.shape[1] - 1:
        neighbor = i, j + 1
        if not grid[neighbor]:
            yield neighbor

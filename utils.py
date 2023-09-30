from ...constants import RIGHT, UP, LEFT, DOWN


def is_on_grid(pos, grid_size):
    return 0 <= pos[0] < grid_size[0] and 0 <= pos[1] < grid_size[1]


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

from collections import defaultdict

import numpy as np
import pytest

from .dijkstra import dijkstra, print_array
from .evaluation_functions import prefer_eating, prefer_battle, calculate_voronoy_areas
from .search_functions import _negamax_moves, _negamax_ab_moves
from .snake import FastSnake
from ...constants import Move


def round_values(moves):
    for move in moves:
        moves[move] = round(moves[move])


class TestCorner:
    grid_size = (3, 3)
    """
    It's player 0 turn. If you move into corner, you will die
    |0    |
    |0    |
    |* 1 1|
    """
    player = FastSnake(id=0, positions=np.array([
        [0, 1],
        [0, 2],
    ]))
    opponent = FastSnake(id=1, positions=np.array([
        [1, 0],
        [2, 0],
    ]))
    candies = [np.array([0, 0])]

    @pytest.mark.parametrize('search_function', [_negamax_moves, _negamax_ab_moves])
    def test_depth_0(self, search_function):
        # at depth 0 prefer to move into the corner to eat
        moves = dict(search_function(self.grid_size, self.player, self.opponent, self.candies, 0))
        round_values(moves)
        print(moves)
        assert moves[Move.UP] == -99
        assert moves[Move.DOWN] == 1
        assert Move.LEFT not in moves
        assert moves[Move.RIGHT] == 0

    @pytest.mark.parametrize('search_function', [_negamax_moves, _negamax_ab_moves])
    def test_depth_1(self, search_function):
        # At depth 1, moving to the corner will trap player 1
        moves = dict(search_function(self.grid_size, self.player, self.opponent, self.candies, 1))
        round_values(moves)
        print(moves)
        assert moves[Move.UP] == -99
        assert moves[Move.DOWN] == -99
        assert Move.LEFT not in moves
        assert moves[Move.RIGHT] == -1

    @pytest.mark.parametrize('search_function', [_negamax_moves, _negamax_ab_moves])
    def test_depth_2(self, search_function):
        moves = dict(search_function(self.grid_size, self.player, self.opponent, self.candies, 2))
        round_values(moves)
        print(moves)
        assert moves[Move.UP] == -99
        assert moves[Move.DOWN] == -99
        assert Move.LEFT not in moves
        assert moves[Move.RIGHT] == -1


@pytest.mark.skip
def test_minimax_avoid_dead_ends():
    grid_size = (3, 6)
    """
    Player 0 shouldn't move to the top, because it can't reach its tail
    |     |
    |0 0 0|
    |0    |
    |0    |
    |     |
    |1 1 1|
    """
    player = FastSnake(id=0, positions=np.array([
        [2, 4],
        [1, 4],
        [0, 4],
        [0, 3],
        [0, 2],
    ]))
    opponent = FastSnake(id=1, positions=np.array([
        [0, 0],
        [1, 0],
        [2, 0],
    ]))
    candies = []
    moves = dict(_negamax_moves(grid_size, player, opponent, candies, 0))
    # round_values(moves)
    print(moves)
    assert moves[Move.DOWN] > moves[Move.UP] > moves[Move.LEFT]


class TestFastWin:
    grid_size = (3, 3)
    """
    Player 0 should move down because it guarantees a win faster
    |0 0  |
    |1 0  |
    |1    |
    """
    player = FastSnake(id=0, positions=np.array([
        [1, 1],
        [1, 2],
        [0, 2],
    ]))
    opponent = FastSnake(id=1, positions=np.array([
        [0, 0],
        [0, 1],
    ]))
    candies = []

    @pytest.mark.parametrize('search_function', [_negamax_moves, _negamax_ab_moves])
    @pytest.mark.parametrize('evaluation_function', [prefer_eating, prefer_battle])
    def test_depth_0(self, search_function, evaluation_function):
        moves = dict(search_function(self.grid_size, self.player, self.opponent, self.candies, 0, evaluation_function))
        # round_values(moves)
        print()
        print(moves)
        assert moves[Move.DOWN] > moves[Move.RIGHT] > moves[Move.LEFT] == moves[Move.UP]

    @pytest.mark.parametrize('search_function', [_negamax_moves, _negamax_ab_moves])
    @pytest.mark.parametrize('evaluation_function', [prefer_eating, prefer_battle])
    def test_depth_1(self, search_function, evaluation_function):
        moves = dict(
            search_function(self.grid_size, self.player, self.opponent, self.candies, 1, evaluation_function))
        # round_values(moves)
        print()
        print(moves)
        # At depth 1 the bot should see a win in 1 move
        assert moves[Move.DOWN] > moves[Move.RIGHT] > moves[Move.LEFT] == moves[Move.UP]

    @pytest.mark.parametrize('search_function', [_negamax_moves, _negamax_ab_moves])
    @pytest.mark.parametrize('evaluation_function', [prefer_eating, prefer_battle])
    def test_depth_2(self, search_function, evaluation_function):
        moves = dict(
            search_function(self.grid_size, self.player, self.opponent, self.candies, 2, evaluation_function))
        # round_values(moves)
        print()
        print(moves)
        assert moves[Move.DOWN] == moves[Move.RIGHT] > moves[Move.LEFT] == moves[Move.UP]


def test_minimax_candies():
    grid_size = (10, 3)
    """
    It's player 0 turn. It should move towards the candy
    |0 0        * |
    |             |
    |1 1          |
    """
    player = FastSnake(id=0, positions=np.array([
        [1, 2],
        [0, 2],
    ]))
    opponent = FastSnake(id=1, positions=np.array([
        [1, 0],
        [0, 0],
    ]))
    candies = [np.array([9, 2])]
    moves = dict(_negamax_moves(grid_size, player, opponent, candies, 0))

    assert 1 > moves[Move.RIGHT] > moves[Move.DOWN] > 0


def test_minimax_closest_candies():
    grid_size = (10, 3)
    """
    It's player 0 turn. It only move towards the candies that it can reach earlier
    |0 0 0      * |
    |             |
    |1 1 *        |
    """
    print()
    player = FastSnake(id=0, positions=np.array([
        [2, 2],
        [1, 2],
        [0, 2],
    ]))
    opponent = FastSnake(id=1, positions=np.array([
        [1, 0],
        [0, 0],
    ]))
    candies = [np.array([9, 2]), np.array([2, 0])]
    moves = dict(_negamax_moves(grid_size, player, opponent, candies, 0))
    print(moves)
    assert 1 > moves[Move.RIGHT] > moves[Move.DOWN] > 0


def test_minimax_candy_on_body():
    grid_size = (10, 3)
    """
    It's player 0 turn. It should NOT move towards the candy
    |0 0  |
    |     |
    |1 *  |
    """
    player = FastSnake(id=0, positions=np.array([
        [1, 2],
        [0, 2],
    ]))
    opponent = FastSnake(id=1, positions=np.array([
        [1, 0],
        [0, 0],
    ]))
    candies = [np.array([1, 0])]
    moves = dict(_negamax_moves(grid_size, player, opponent, candies, 0))
    print(moves)
    assert moves[Move.DOWN] == moves[Move.RIGHT] > moves[Move.LEFT]


def test_bot_prefers_to_be_close():
    grid_size = (3, 4)
    """
    Snake 0 is very long, it should go towards the opponent
    |0 0  |
    |     |
    |     |
    |1 1  |
    """
    player = FastSnake(id=0, positions=np.vstack(([1, 3], np.tile([0, 3], (20, 1)))))
    opponent = FastSnake(id=1, positions=np.vstack(([1, 0], np.tile([0, 0], (20, 1)))))
    candies = []
    moves = dict(_negamax_moves(grid_size, player, opponent, candies, 0, prefer_battle))
    print(moves)
    assert moves[Move.DOWN] > moves[Move.RIGHT] > moves[Move.LEFT]


@pytest.mark.parametrize('evaluation_function', [prefer_eating, prefer_battle])
def test_suicide(evaluation_function):
    grid_size = (5, 3)
    """
    It's player 0 turn. If you suicide you win
    |0 0 0 0 0 |
    |          |
    |1 1       |
    """
    player = FastSnake(id=0, positions=np.array([
        [4, 2],
        [3, 2],
        [2, 2],
        [1, 2],
        [0, 2],
    ]))
    opponent = FastSnake(id=1, positions=np.array([
        [1, 0],
        [0, 0],
    ]))
    candies = []

    moves = dict(_negamax_moves(grid_size, player, opponent, candies, 0, evaluation_function))
    print(moves)

    assert moves[Move.LEFT] == 99
    assert moves[Move.LEFT] > moves[Move.DOWN]


@pytest.mark.parametrize('evaluation_function', [prefer_eating, prefer_battle])
def test_dont_suicide(evaluation_function):
    grid_size = (5, 3)
    """
    It's player 0 turn. If you suicide you DRAW
    |0 0 0 0   |
    |          |
    |1 1       |
    """
    player = FastSnake(id=0, positions=np.array([
        [3, 2],
        [2, 2],
        [1, 2],
        [0, 2],
    ]))
    opponent = FastSnake(id=1, positions=np.array([
        [1, 0],
        [0, 0],
    ]))
    candies = []

    moves = dict(_negamax_moves(grid_size, player, opponent, candies, 0, evaluation_function))
    print(moves)

    assert moves[Move.LEFT] == -50
    assert moves[Move.DOWN] > moves[Move.LEFT]
    assert moves[Move.RIGHT] > moves[Move.LEFT]


def test_voronoy_heuristic():
    grid_size = (3, 3)
    """
    |x   |
    |    |
    |0  1|
    """
    collision_grid = np.zeros(grid_size, dtype=bool)
    collision_grid[0, 2] = True
    player_dist = dijkstra((0, 0), collision_grid)
    opponent_dist = dijkstra((2, 0), collision_grid)

    player_first, opponent_first = calculate_voronoy_areas(player_dist, opponent_dist)

    print()
    print_array(player_dist)
    print_array(opponent_dist)
    print_array(player_first)
    print_array(opponent_first)


def test_voronoy_heuristic2():
    grid_size = (4, 3)
    """
    |       |
    |       |
    |0 - - 1|
    """
    collision_grid = np.zeros(grid_size, dtype=bool)
    player_dist = dijkstra((0, 0), collision_grid)
    opponent_dist = dijkstra((3, 0), collision_grid)

    player_first, opponent_first = calculate_voronoy_areas(player_dist, opponent_dist)

    print()
    print_array(player_dist)
    print_array(opponent_dist)
    print_array(player_first)
    print_array(opponent_first)


def test_voronoy_heuristic_distinct():
    grid_size = (3, 3)
    """
    |  x  |
    |  x  |
    |0 x 1|
    """
    print()
    collision_grid = np.zeros(grid_size, dtype=bool)
    collision_grid[(1, 1, 1), (0, 1, 2)] = True
    player_dist = dijkstra((0, 0), collision_grid)
    opponent_dist = dijkstra((2, 0), collision_grid)

    player_first, opponent_first = calculate_voronoy_areas(player_dist, opponent_dist)

    print_array(player_dist)
    print_array(opponent_dist)
    print_array(player_first)
    print_array(opponent_first)


@pytest.mark.skip
def test_traveling_salesman():
    grid_size = (3, 3)
    """
    |c c 1|
    |     |
    |0   c|
    """
    player_head = (0, 0)
    opponent_head = (2, 2)
    collision_grid = np.zeros(grid_size, dtype=bool)
    player_dist = dijkstra(player_head, collision_grid)
    opponent_dist = dijkstra(opponent_head, collision_grid)
    candies = dict(enumerate([(0, 2), (1, 2), (2, 0)]))

    print()
    distance_matrix = defaultdict(dict)
    for i in range(len(candies)):
        for j in range(len(candies)):
            if i != j:
                ci = candies[i]
                cj = candies[j]
                distance_matrix[i][j] = abs(ci[0] - cj[0]) + abs(ci[1] - cj[1])

    for ci, c in candies.items():
        distance_matrix[-1][ci] = player_dist[c]
    for ci, c in candies.items():
        distance_matrix[-2][ci] = player_dist[c]

    print(distance_matrix)

    player_pos = -1
    opponent_pos = -2
    player_walked = 0
    opponent_walked = 0
    player_candy_order = [0, 1, 2]
    opponent_candy_order = [0, 1, 2]
    player_candies_eaten = 0
    opponent_candies_eaten = 0
    while True:
        player_candy_distance = distance_matrix[player_pos][player_candy_order[0]]
        opponent_candy_distance = distance_matrix[opponent_pos][opponent_candy_order[0]]
        print(player_candy_distance, opponent_candy_distance)
        if player_candy_distance <= opponent_candy_distance:
            print('player eat candy')
            player_pos = player_candy_order[0]
            player_candy_order.pop(0)
            opponent_walked += player_candy_distance
            player_candies_eaten += 1
        else:
            print('opponent eats candy')
            opponent_pos = opponent_candy_order[0]
            player_candy_order.pop(0)
            player_walked += opponent_candy_distance
            opponent_candies_eaten += 1

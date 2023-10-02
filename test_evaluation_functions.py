import numpy as np
import pytest

from .evaluation_functions import prefer_eating, prefer_battle
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


@pytest.mark.skip
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
    moves = dict(_negamax_moves(grid_size, player, opponent, candies, 0))
    print(moves)
    assert moves[Move.DOWN] > moves[Move.RIGHT] > moves[Move.LEFT]


def test_suicide():
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

    moves = dict(_negamax_moves(grid_size, player, opponent, candies, 0))
    print(moves)

    assert moves[Move.LEFT] == 99
    assert moves[Move.LEFT] > moves[Move.DOWN]


def test_dont_suicide():
    grid_size = (5, 3)
    """
    It's player 0 turn. If you suicide you win
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

    moves = dict(_negamax_moves(grid_size, player, opponent, candies, 0))
    print(moves)

    assert moves[Move.LEFT] == -50
    assert moves[Move.DOWN] > moves[Move.LEFT]

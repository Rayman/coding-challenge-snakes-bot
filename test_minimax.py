import numpy as np

from .minimax import moves_with_scores
from ...constants import Move
from ...snake import Snake


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
    player = Snake(id=0, positions=np.array([
        [0, 1],
        [0, 2],
    ]))
    opponent = Snake(id=1, positions=np.array([
        [1, 0],
        [2, 0],
    ]))
    candies = [np.array([0, 0])]

    def test_depth_0(self):
        # at depth 0 prefer to move into the corner to eat
        moves = dict(moves_with_scores(self.grid_size, self.player, self.opponent, self.candies, 0))
        round_values(moves)
        print(moves)
        assert moves[Move.UP] == -99
        assert moves[Move.DOWN] == 1
        assert Move.LEFT not in moves
        assert moves[Move.RIGHT] == 0

    def test_depth_1(self):
        # at depth 1 moving to the corner result in the other player growing
        moves = dict(moves_with_scores(self.grid_size, self.player, self.opponent, self.candies, 1))
        round_values(moves)
        print(moves)
        assert moves[Move.UP] == -99
        assert moves[Move.DOWN] == 1
        assert Move.LEFT not in moves
        assert moves[Move.RIGHT] == -1

    def test_depth_2(self):
        # at depth 2 moving to the corner results in death
        moves = dict(moves_with_scores(self.grid_size, self.player, self.opponent, self.candies, 2))
        round_values(moves)
        print(moves)
        assert moves[Move.UP] == -99
        assert moves[Move.DOWN] == -99
        assert Move.LEFT not in moves
        assert moves[Move.RIGHT] == -1


def test_minimax_candies():
    grid_size = (10, 3)
    """
    It's player 0 turn. It should move towards the candy
    |0 0        * |
    |             |
    |1 1          |
    """
    player = Snake(id=0, positions=np.array([
        [1, 2],
        [0, 2],
    ]))
    opponent = Snake(id=1, positions=np.array([
        [1, 0],
        [0, 0],
    ]))
    candies = [np.array([9, 2])]
    moves = dict(moves_with_scores(grid_size, player, opponent, candies, 0))

    assert 1 > moves[Move.RIGHT] > moves[Move.DOWN] > 0


def test_suicide():
    grid_size = (5, 3)
    """
    It's player 0 turn. If you suicide you win
    |0 0 0 0 0 |
    |          |     
    |1 1       | 
    """
    player = Snake(id=0, positions=np.array([
        [4, 2],
        [3, 2],
        [2, 2],
        [1, 2],
        [0, 2],
    ]))
    opponent = Snake(id=1, positions=np.array([
        [1, 0],
        [0, 0],
    ]))
    candies = []

    moves = dict(moves_with_scores(grid_size, player, opponent, candies, 0))
    print(moves)

    assert moves[Move.LEFT] == 99
    assert moves[Move.LEFT] > moves[Move.DOWN]

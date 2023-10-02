from random import choice
from typing import Tuple

from .evaluation_functions import prefer_eating, prefer_battle
from .search_functions import negamax_moves
from .snake import FastSnake
from ...bot import Bot
from ...constants import Move

__all__ = ['Slifer']


class Slifer(Bot):
    def __init__(self, id: int, grid_size: Tuple[int, int], depth=0):
        super().__init__(id=id, grid_size=grid_size)
        self.battle_mode = False
        self.depth = depth

    @property
    def name(self):
        return 'Slifer the Sky Dragon'

    @property
    def contributor(self):
        return 'Rayman'

    def determine_next_move(self, snake, other_snakes, candies) -> Move:
        player = FastSnake(id=snake.id, positions=snake.positions)
        opponent = FastSnake(id=other_snakes[0].id, positions=other_snakes[0].positions)

        max_score = float('-inf')
        moves = []

        if len(player) > len(opponent) > 10:
            self.battle_mode = True

        if self.battle_mode:
            evaluation_function = prefer_battle
        else:
            evaluation_function = prefer_eating

        for move, score in negamax_moves(self.grid_size, player, opponent, candies, self.depth,
                                         evaluation_function):
            if score > max_score:
                max_score = score
                moves = [move]
            elif score == max_score:
                moves.append(move)
        return choice(moves)


class Slifer0(Slifer):
    def __init__(self, id: int, grid_size: Tuple[int, int]):
        super().__init__(id=id, grid_size=grid_size, depth=0)

    @property
    def name(self):
        return 'Slifer0'


class Slifer1(Slifer):
    def __init__(self, id: int, grid_size: Tuple[int, int]):
        super().__init__(id=id, grid_size=grid_size, depth=1)

    @property
    def name(self):
        return 'Slifer1'


class Slifer2(Slifer):
    def __init__(self, id: int, grid_size: Tuple[int, int]):
        super().__init__(id=id, grid_size=grid_size, depth=2)

    @property
    def name(self):
        return 'Slifer2'

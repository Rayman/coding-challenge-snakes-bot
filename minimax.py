from random import choice
from typing import Tuple

import numpy as np

from .board import Node
from .evaluation_functions import prefer_eating, prefer_battle
from .search_functions import negamax
from ...bot import Bot
from ...constants import Move

__all__ = ['MiniMax']


def moves_with_scores(grid_size, player, opponent, candies, depth, evaluation_function=None):
    evaluation_function = prefer_eating if evaluation_function is None else evaluation_function
    node = Node(grid_size, player, opponent, candies)
    for child in node.children():
        move = move_to_enum(child.opponent[0] - player[0])
        # print(f'evaluating move={move}')
        value = -negamax(child, depth, evaluation_function)
        # print(f'evaluation result for move={move} value={value}\n')
        yield move, value


def is_on_grid(pos, grid_size):
    return 0 <= pos[0] < grid_size[0] and 0 <= pos[1] < grid_size[1]


def collides(head, snake):
    for segment in snake:
        if np.array_equal(head, segment):
            return True
    return False


def move_to_enum(move: np.array) -> Move:
    if move[0] == 0:
        if move[1] == 1:
            return Move.UP
        else:
            assert move[1] == -1, f'Unexpected move={move}'
            return Move.DOWN
    else:
        assert move[1] == 0, f'Unexpected move={move}'
        if move[0] == 1:
            return Move.RIGHT
        else:
            assert move[0] == -1, f'Unexpected move={move}'
            return Move.LEFT


class MiniMax(Bot):
    def __init__(self, id: int, grid_size: Tuple[int, int]):
        self.grid_size = grid_size
        self.battle_mode = False

    @property
    def name(self):
        return 'Slifer the Sky Dragon'

    @property
    def contributor(self):
        return 'Rayman'

    def determine_next_move(self, snake, other_snakes, candies) -> Move:
        player = snake
        opponent = other_snakes[0]

        max_score = float('-inf')
        moves = []

        if len(player) > len(opponent) > 10:
            self.battle_mode = True

        if self.battle_mode:
            evaluation_function = prefer_battle
        else:
            evaluation_function = prefer_eating

        for move, score in moves_with_scores(self.grid_size, player, opponent, candies, 0, evaluation_function):
            if score > max_score:
                max_score = score
                moves = [move]
            elif score == max_score:
                moves.append(move)
        return choice(moves)

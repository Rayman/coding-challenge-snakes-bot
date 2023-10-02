from copy import deepcopy
from typing import List

import numpy as np

from .snake import FastSnake
from ...constants import MOVE_VALUE_TO_DIRECTION


class Node:
    """
    Game state just before player makes a move
    """

    def __init__(self, grid_size, player: FastSnake, opponent: FastSnake, candies: List[np.array]):
        assert isinstance(player, FastSnake)
        assert isinstance(opponent, FastSnake)
        self.grid_size = grid_size
        self.player = player
        self.opponent = opponent
        self.candies = candies

    def children(self):
        for move in MOVE_VALUE_TO_DIRECTION.values():
            player = deepcopy(self.player)

            next_head = player[0] + move
            if not (0 <= next_head[0] < self.grid_size[0] and 0 <= next_head[1] < self.grid_size[1]):
                # Not a legal move, so not returning a Terminal node
                continue

            for i, candy in enumerate(self.candies):
                if np.array_equal(player[0] + move, candy):
                    player.move(move, grow=True)
                    break
            else:
                player.move(move)

            if player[0] in self.player[1:]:
                # print(f'snake {player.id} collided with itself')
                yield TerminalNode(self.opponent, player)
                continue
            if self.opponent.collides(player[0]):
                # print(f'snake {player.id} collided with the opponent')
                yield TerminalNode(self.opponent, player)
                continue
            yield self.__class__(self.grid_size, self.opponent, player, self.candies)

    def __str__(self):
        grid = np.empty(self.grid_size, dtype=str)
        for x in range(self.grid_size[0]):
            for y in range(self.grid_size[1]):
                grid[x, y] = ' '
        for candy in self.candies:
            grid[candy[0], candy[1]] = '*'
        # if the player died, don't print it
        for pos in self.player:
            grid[pos[0], pos[1]] = 'P'
        for pos in self.opponent:
            grid[pos[0], pos[1]] = 'O'
        return str(np.flipud(grid.T))


class TerminalNode:
    """
    Game state where the opponent has just died
    """

    def __init__(self, player, opponent):
        self.player = player
        self.opponent = opponent

    def __str__(self):
        return f'[[ player {self.player.id} has survived ]]'

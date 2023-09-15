from copy import deepcopy
from random import choice
from typing import List

import numpy as np

from .shortest_path import dijkstra, neighbors
from ...bot import Bot
from ...constants import MOVE_VALUE_TO_DIRECTION, Move
from ...snake import Snake


def moves_with_scores(grid_size, player, opponent, candies, depth):
    node = Node(grid_size, player, opponent, candies)
    for child in node.children():
        move = move_to_enum(child.opponent[0] - player[0])
        # print(f'evaluating move={move}')
        value = -negamax(child, depth)
        # print(f'evaluation result for move={move} value={value}')
        yield move, value


def negamax(node, depth):
    # print(f'depth={depth} player={node.player.id}')
    value: float
    if depth == 0 or isinstance(node, TerminalNode):
        value = node.heuristic_value()
        # print(f'terminal node with value={value}\n{node}')
    else:
        values = []
        for child in node.children():
            values.append(-negamax(child, depth - 1))
        # print(f'values={values}')
        value = max(values)
    # print(f'depth={depth} player={node.player.id} value={value:3}')
    return value


class TerminalNode:
    """
    Game state where the current player can't move anymore because the opponent died
    """

    def __init__(self, player, opponent):
        self.player = player
        self.opponent = opponent

    def heuristic_value(self):
        player_score = len(self.player) * 2
        opponent_score = len(self.opponent)
        if player_score > opponent_score:
            return 99
        elif player_score < opponent_score:
            return -99
        else:
            return 50

    def __str__(self):
        return f'[[ player {self.player.id} has won ]]'


class Node:
    """
    Game state just before player makes a move
    """

    def __init__(self, grid_size, player: Snake, opponent: Snake, candies: List[np.array]):
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

            self_collision = False
            # self collision, don't check head
            for p in self.player[1:]:
                if np.array_equal(p, player[0]):
                    self_collision = True
                    break
            if self_collision:
                # print(f'snake {player.id} collided with itself')
                yield TerminalNode(self.opponent, player)
                continue
            if self.opponent.collides(player[0]):
                # print(f'snake {player.id} collided with the opponent')
                yield TerminalNode(self.opponent, player)
                continue
            yield Node(self.grid_size, self.opponent, player, self.candies)

    def heuristic_value(self):
        collision_grid = np.zeros(self.grid_size, dtype=bool)
        for segment in self.player:
            collision_grid[segment[0], segment[1]] = True
        for segment in self.opponent:
            collision_grid[segment[0], segment[1]] = True

        player_dist = dijkstra(self.player[0], collision_grid)
        # print('player:\n', np.flipud(player_dist.T))

        opponent_dist = dijkstra(self.opponent[0], collision_grid)
        # print('opponent:\n', np.flipud(opponent_dist.T))

        length_difference = len(self.player) - len(self.opponent)
        candy_bonus = self.candy_bonus(player_dist, opponent_dist)
        # tail_penalty = self.tail_penalty(collision_grid, player_dist, opponent_dist)

        return length_difference + 0.01 * candy_bonus
        # return length_difference

    def candy_bonus(self, player_dist, opponent_dist):
        distance_player_candy = self._distance_to_candy(player_dist)
        distance_opponent_candy = self._distance_to_candy(opponent_dist)
        player_candy_bonus = -min(40, distance_player_candy)
        opponent_candy_bonus = -min(40, distance_opponent_candy)

        return player_candy_bonus - opponent_candy_bonus

    def tail_penalty(self, collision_grid, player_dist, opponent_dist):
        max_int = np.iinfo(player_dist.dtype).max
        player_tail_dist = min((player_dist[n[0], n[1]] for n in neighbors(self.player[-1], collision_grid)),
                               default=max_int)
        opponent_tail_dist = min((opponent_dist[n[0], n[1]] for n in neighbors(self.opponent[-1], collision_grid)),
                                 default=max_int)

        player_tail_penalty = 30 if player_tail_dist == max_int else 0
        opponent_tail_penalty = 30 if opponent_tail_dist == max_int else 0

        # print(f'tail penalty: player={player_tail_penalty} opponent={opponent_tail_penalty}')
        return player_tail_penalty - opponent_tail_penalty

    def _distance_to_candy(self, dist: np.array):
        if not self.candies:
            return 0

        return min(dist[candy[0], candy[1]] for candy in self.candies)

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
    """
    Pick a random move, given that it is collision free
    """

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
        for move, score in moves_with_scores(self.grid_size, player, opponent, candies, 0):
            if score > max_score:
                max_score = score
                moves = [move]
            elif score == max_score:
                moves.append(move)
        return choice(moves)

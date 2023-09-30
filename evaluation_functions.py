from typing import List

import numpy as np

from .board import Node
from .board import TerminalNode
from .test_dijkstra import dijkstra
from .utils import neighbors


def prefer_eating(node: Node):
    collision_grid = np.zeros(node.grid_size, dtype=bool)
    for segment in node.player:
        collision_grid[segment[0], segment[1]] = True
    for segment in node.opponent:
        collision_grid[segment[0], segment[1]] = True

    # It's players turn, so if player doesn't have any legal moves left, the opponent has won
    number_of_moves = len(list(neighbors(*node.player[0], collision_grid)))
    if number_of_moves == 0:
        # print(f'Player {node.player.id} has no legal moves available, opponent={node.opponent.id} will survive')
        return -terminal_value(TerminalNode(node.opponent, node.player))
    # We can't yet check how many legal moves the opponent has available, because player still needs to move

    player_dist = dijkstra(node.player[0], collision_grid)
    # print('player:\n', np.flipud(player_dist.T))

    opponent_dist = dijkstra(node.opponent[0], collision_grid)
    # print('opponent:\n', np.flipud(opponent_dist.T))

    length_difference = len(node.player) - len(node.opponent)

    player_candy_bonus = candy_bonus(player_dist, node.candies)  # if len(node.player) < 10 else 0
    opponent_candy_bonus = candy_bonus(opponent_dist, node.candies)  # if len(node.opponent) < 10 else 0

    # print(f'player_opponent_dist={player_opponent_dist}')
    # print(f'length_difference={length_difference} player_candy_bonus={player_candy_bonus} opponent_candy_bonus={opponent_candy_bonus}')
    return length_difference + 0.01 * (player_candy_bonus - opponent_candy_bonus)


def prefer_battle(node: Node):
    collision_grid = np.zeros(node.grid_size, dtype=bool)
    for segment in node.player:
        collision_grid[segment[0], segment[1]] = True
    for segment in node.opponent:
        collision_grid[segment[0], segment[1]] = True

    # It's players turn, so if player doesn't have any legal moves left, the opponent has won
    number_of_moves = len(list(neighbors(*node.player[0], collision_grid)))
    if number_of_moves == 0:
        # print(f'Player {node.player.id} has no legal moves available, opponent={node.opponent.id} will survive')
        return -terminal_value(TerminalNode(node.opponent, node.player))
    # We can't yet check how many legal moves the opponent has available, because player still needs to move

    player_dist = dijkstra(node.player[0], collision_grid)
    # print('player:\n', np.flipud(player_dist.T))

    opponent_dist = dijkstra(node.opponent[0], collision_grid)
    # print('opponent:\n', np.flipud(opponent_dist.T))

    # print(np.flipud((player_dist > opponent_dist).T))
    # print(np.flipud((opponent_dist > player_dist).T))
    voronoy_heuristic = np.count_nonzero(player_dist < opponent_dist) - np.count_nonzero(
        opponent_dist < player_dist)

    max_int = np.iinfo(player_dist.dtype).max
    player_opponent_dist = min((player_dist[n[0], n[1]] for n in neighbors(*node.opponent[0], collision_grid)),
                               default=max_int)
    player_opponent_dist = min(*node.grid_size, player_opponent_dist)

    tail_heuristic = tail_penalty(node, collision_grid, player_dist, opponent_dist)

    # print(f'voronoy_heuristic={voronoy_heuristic} tail_penalty={tail_penalty}')
    return voronoy_heuristic / node.grid_size[0] / node.grid_size[1]  # + tail_penalty


def terminal_value(node: TerminalNode) -> float:
    """
    Heuristic value of a game that is finished
    """
    player_score = len(node.player) * 2
    opponent_score = len(node.opponent)
    # print(f'player {self.player.id} survives player_score={player_score} opponent_score={opponent_score}')
    if player_score > opponent_score:
        return 99
    elif player_score < opponent_score:
        return -99
    else:
        return 50


def tail_penalty(node: Node, collision_grid, player_dist, opponent_dist):
    """

    :param collision_grid:
    :param player_dist:
    :param opponent_dist:
    :return:
    """
    max_int = np.iinfo(player_dist.dtype).max
    player_tail_dist = min((player_dist[n[0], n[1]] for n in neighbors(*node.player[-1], collision_grid)),
                           default=max_int)
    opponent_tail_dist = min((opponent_dist[n[0], n[1]] for n in neighbors(*node.opponent[-1], collision_grid)),
                             default=max_int)

    player_tail_penalty = 30 if player_tail_dist == max_int else 0
    opponent_tail_penalty = 30 if opponent_tail_dist == max_int else 0

    # print(f'tail penalty: player={player_tail_penalty} opponent={opponent_tail_penalty}')
    return player_tail_penalty - opponent_tail_penalty


def candy_bonus(dist, candies: List[np.array]):
    distance_to_candy = _distance_to_candy(dist, candies)
    return -min(40, distance_to_candy)


def _distance_to_candy(dist: np.array, candies: List[np.array]):
    if not candies:
        return 0

    return min(dist[candy[0], candy[1]] for candy in candies)

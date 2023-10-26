from typing import List

import numpy as np
from matplotlib import pyplot as plt

from .board import Node
from .board import TerminalNode
from .dijkstra import dijkstra2, print_array, dijkstra
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

    old_candy_bonus = calculate_candy_bonus(player_dist, opponent_dist, node.candies)

    # player_dist, opponent_dist = calculate_voronoi_diagram(collision_grid, node.player[0], node.opponent[0])
    # print('player:\n', print_array(player_dist))

    # player_dist, opponent_dist = calculate_voronoi_diagram(collision_grid, node.player[0], node.opponent[0])
    # player_dist = player_dist.filled()
    # opponent_dist = opponent_dist.filled()

    length_difference = len(node.player) - len(node.opponent)

    # candy_bonus = calculate_new_candy_bonus(player_dist, opponent_dist, node.candies, node.player[0], node.opponent[0])
    # if candy_bonus != old_candy_bonus:
    #     print(f'candy_bonus={candy_bonus} != old_candy_bonus={old_candy_bonus}')
    #     pass
    # print(f'score={length_difference + 0.01 * candy_bonus} '
    #       f'length_difference={length_difference} candy_bonus={candy_bonus}')
    return length_difference + 0.01 * old_candy_bonus


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

    player_dist, opponent_dist = calculate_voronoi_diagram(collision_grid, node.player[0], node.opponent[0])
    voronoi_heuristic = player_dist.count() - opponent_dist.count()
    player_dist = player_dist.filled()
    opponent_dist = opponent_dist.filled()

    max_int = np.iinfo(player_dist.dtype).max
    player_opponent_dist = min((player_dist[n[0], n[1]] for n in neighbors(*node.opponent[0], collision_grid)),
                               default=max_int)
    player_opponent_dist = min(*node.grid_size, player_opponent_dist)

    tail_heuristic = tail_penalty(node, collision_grid, player_dist, opponent_dist)

    # print(f'voronoi_heuristic={voronoi_heuristic} tail_penalty={tail_penalty}')
    return voronoi_heuristic / node.grid_size[0] / node.grid_size[1]  # + tail_penalty


def calculate_voronoi_areas(player_dist, opponent_dist):
    max = np.iinfo(player_dist.dtype).max

    only_player_reachable = (player_dist != max) & (opponent_dist == max)
    only_opponent_reachable = (player_dist == max) & (opponent_dist != max)
    both_reachable = (player_dist != max) & (opponent_dist != max)

    player_first = (player_dist * 2 < opponent_dist * 2 + 1) & (player_dist != max)
    opponent_first = (opponent_dist * 2 + 1 < player_dist * 2) & (opponent_dist != max)

    player_first = (both_reachable & player_first) | only_player_reachable
    opponent_first = (both_reachable & opponent_first) | only_opponent_reachable

    # mat = np.zeros(player_dist.shape, dtype=player_dist.dtype)
    # mat[player_first] = player_dist[player_first]
    # mat[opponent_first] = - opponent_dist[opponent_first]
    # print_array(mat)

    return player_first, opponent_first


def calculate_voronoi_diagram(collision_grid, player_head, opponent_head):
    voronoi = dijkstra2(((player_head, 0), (opponent_head, 1)), collision_grid)
    # print(f'voronoi=\n{array2str(voronoi)}')
    free = ~collision_grid
    player_first = ~(voronoi & 1).astype(bool) & free
    opponent_first = (voronoi & 1).astype(bool) & free
    # print(f'player_first=\n{array2str(player_first)}')
    # print(f'opponent_first=\n{array2str(opponent_first)}')
    max_int = np.iinfo(voronoi.dtype).max
    player_dist = np.ma.masked_array(voronoi, ~player_first, fill_value=max_int)
    opponent_dist = np.ma.masked_array(voronoi, ~opponent_first, fill_value=max_int)
    return player_dist, opponent_dist


ax = None


def plot_voronoi_heuristic(node: Node):
    collision_grid = np.zeros(node.grid_size, dtype=bool)
    for segment in node.player:
        collision_grid[segment[0], segment[1]] = True
    for segment in node.opponent:
        collision_grid[segment[0], segment[1]] = True

    player_dist, opponent_dist = calculate_voronoi_diagram(collision_grid, node.player[0], node.opponent[0])
    mat = np.zeros(node.grid_size, dtype=player_dist.dtype)
    mat[~player_dist.mask] = 8 + player_dist[~player_dist.mask]
    mat[~opponent_dist.mask] = -8 - opponent_dist[~opponent_dist.mask]
    mat = np.flipud(mat.T)

    print_array(mat)
    global ax
    if not ax:
        ax = plt.matshow(mat)
    else:
        ax.set_data(mat)
    plt.draw()
    plt.pause(1e-6)


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


def calculate_candy_bonus(player_dist, opponent_dist, candies: List[np.array]):
    if not candies:
        return 0
    # print('calculate_old_candy_bonus')
    player_candy_index = min(range(len(candies)), key=lambda i: player_dist[tuple(candies[i])])
    opponent_candy_index = min(range(len(candies)), key=lambda i: opponent_dist[tuple(candies[i])])
    # print(f'original candy assignment: player={player_candy_index},d={player_dist[tuple(candies[player_candy_index])]} opponent={opponent_candy_index},d={opponent_dist[tuple(candies[opponent_candy_index])]}')
    if player_candy_index == opponent_candy_index:
        if player_dist[tuple(candies[player_candy_index])] > opponent_dist[tuple(candies[opponent_candy_index])]:
            # print('player has to choose a different candy')
            player_candy_index = min((i for i in range(len(candies)) if i != opponent_candy_index),
                                     key=lambda i: player_dist[tuple(candies[i])], default=-1)
        else:
            # print('opponent has to choose a different candy')
            opponent_candy_index = min((i for i in range(len(candies)) if i != player_candy_index),
                                       key=lambda i: opponent_dist[tuple(candies[i])], default=-1)
    # print(f'resolution: player={player_candy_index},d={player_dist[tuple(candies[player_candy_index])]} opponent={opponent_candy_index},d={opponent_dist[tuple(candies[opponent_candy_index])]}')
    player_candy_bonus = player_dist[tuple(candies[player_candy_index])] if player_candy_index != -1 else 40
    opponent_candy_bonus = opponent_dist[tuple(candies[opponent_candy_index])] if opponent_candy_index != -1 else 40
    # print(f'new player_candy_bonus={player_candy_bonus} opponent_candy_bonus={opponent_candy_bonus}')
    cb = opponent_candy_bonus - player_candy_bonus
    return min(40, max(-40, cb))


def calculate_new_candy_bonus(player_dist, opponent_dist, candies: List[np.array], player_head, opponent_head):
    if not candies:
        return 0
    # print('calculate_candy_bonus')
    player_candy_index = min(range(len(candies)), key=lambda i: player_dist[tuple(candies[i])])
    opponent_candy_index = min(range(len(candies)), key=lambda i: opponent_dist[tuple(candies[i])])
    # print(f'original candy assignment: player={player_candy_index},d={player_dist[tuple(candies[player_candy_index])]} opponent={opponent_candy_index},d={opponent_dist[tuple(candies[opponent_candy_index])]}')

    candy_bonus = 0
    int_max = np.iinfo(player_dist.dtype).max

    if player_dist[tuple(candies[player_candy_index])] != int_max:
        player_candy_bonus = 40 - player_dist[tuple(candies[player_candy_index])]
    else:
        # all candies are closer to the other player. Choose another candy, still a chance to eat it
        print("player can't reach all candies")
        player_candy_bonus = 16 - min((np.linalg.norm(candies[i] - player_head)
                                       for i in range(len(candies)) if i != opponent_candy_index), default=16)

    if opponent_dist[tuple(candies[opponent_candy_index])] != int_max:
        opponent_candy_bonus = 40 - opponent_dist[tuple(candies[opponent_candy_index])]
    else:
        # all candies are closer to the other player. Choose another candy, still a chance to eat it
        print("opponent can't reach all candies")
        opponent_candy_bonus = 16 - min((np.linalg.norm(candies[i] - opponent_head)
                                         for i in range(len(candies)) if i != player_candy_index), default=16)
    print(f'player_candy_bonus={player_candy_bonus} opponent_candy_bonus={opponent_candy_bonus}')
    return player_candy_bonus - opponent_candy_bonus

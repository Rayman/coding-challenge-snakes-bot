from math import inf
from random import choice, shuffle
from typing import Callable, List

import numpy as np

from .board import TerminalNode, Node
from .evaluation_functions import terminal_value, prefer_eating
from ...constants import Move


def negamax(node, depth, evaluation_function: Callable):
    # print(f'depth={depth} player={node.player.id}')
    value: float

    if isinstance(node, TerminalNode):
        # print(f'terminal_value={terminal_value(node)}\n{node}')
        return terminal_value(node)
    elif depth == 0:
        value = evaluation_function(node)
        # print(f'heuristic value={value}\n{node}')
    else:
        values = []
        for child in node.children():
            values.append(-negamax(child, depth - 1, evaluation_function))
        # print(f'values={values}')
        value = max(values)
    # print(f'depth={depth} player={node.player.id} value={value:3}')
    return value


def negamax_move(grid_size, player, opponent, candies, depth, evaluation_function=None):
    best_value = -inf
    best_moves = []
    for move, value in _negamax_moves(grid_size, player, opponent, candies, depth,
                                      evaluation_function):
        if value > best_value:
            best_value = value
            best_moves = [move]
        elif value == best_value:
            best_moves.append(move)
    return choice(best_moves)


def _negamax_moves(grid_size, player, opponent, candies, depth, evaluation_function=None):
    evaluation_function = prefer_eating if evaluation_function is None else evaluation_function
    node = Node(grid_size, player, opponent, candies)
    for child in move_ordering(list(node.children())):
        move = move_to_enum((child.opponent[0][0] - player[0][0], child.opponent[0][1] - player[0][1]))
        # print(f'evaluating move={move}')
        value = -negamax(child, depth, evaluation_function)
        # print(f'evaluation result for move={move} value={value}\n')
        yield move, value


def negamax_ab(node, depth, a, b, evaluation_function: Callable):
    # print(f'depth={depth} player={node.player.id} a={a} b={b}')
    value: float

    if isinstance(node, TerminalNode):
        # print(f'terminal_value={terminal_value(node)}\n{node}')
        return terminal_value(node)
    elif depth == 0:
        value = evaluation_function(node)
        # print(f'heuristic value={value}\n{node}')
    else:
        value = -inf
        for child in move_ordering(list(node.children())):
            move = move_to_enum((child.opponent[0][0] - node.player[0][0], child.opponent[0][1] - node.player[0][1]))
            # print(f'evaluating child move={move}')
            value = max(value, -negamax_ab(child, depth - 1, -b, -a, evaluation_function))
            a = max(a, value)
            if a >= b:
                # print(f'ab cut-off at depth={depth} value={value} a={a} b={b}\n{node}')
                break
    # print(f'depth={depth} player={node.player.id} value={value:3}')
    return value


def negamax_ab_move(grid_size, player, opponent, candies, depth, evaluation_function=None):
    evaluation_function = prefer_eating if evaluation_function is None else evaluation_function
    node = Node(grid_size, player, opponent, candies)
    best_value = -inf
    best_move = None
    a = -inf  # lower bound on
    b = inf
    for child in move_ordering(list(node.children())):
        move = move_to_enum((child.opponent[0][0] - player[0][0], child.opponent[0][1] - player[0][1]))
        # print(f'evaluating move={move}')
        value = -negamax_ab(child, depth, -b, -a, evaluation_function)
        # print(f'evaluation result for move={move} value={value} \n')
        a = max(a, value)
        if value > best_value:
            best_value = value
            best_move = move
    return best_move


def _negamax_ab_moves(grid_size, player, opponent, candies, depth, evaluation_function=None):
    evaluation_function = prefer_eating if evaluation_function is None else evaluation_function
    node = Node(grid_size, player, opponent, candies)
    for child in move_ordering(list(node.children())):
        move = move_to_enum((child.opponent[0][0] - player[0][0], child.opponent[0][1] - player[0][1]))
        # print(f'evaluating move={move}')
        value = -negamax_ab(child, depth, -inf, inf, evaluation_function)
        # print(f'evaluation result for move={move} value={value} \n')
        yield move, value


def move_ordering(children: List[Node]) -> List[Node]:
    winning = []
    normal = []
    losing = []

    for node in children:
        if isinstance(node, TerminalNode):
            value = terminal_value(node)
            if value < 0:
                winning.append(node)
            elif value > 0:
                losing.append(node)
        else:
            normal.append(node)

    shuffle(winning)
    # print('yielding winning moves')
    yield from winning
    shuffle(normal)
    # print('yielding normal moves')
    yield from normal
    shuffle(losing)
    # print('yielding losing moves')
    yield from losing


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

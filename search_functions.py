from typing import Callable

from .board import TerminalNode
from .evaluation_functions import terminal_value


def negamax(node, depth, evaluation_function: Callable):
    # print(f'depth={depth} player={node.player.id}')
    value: float

    if isinstance(node, TerminalNode):
        return terminal_value(node)
    elif depth == 0:
        value = evaluation_function(node)
        # print(f'terminal node with value={value}\n{node}')
    else:
        values = []
        for child in node.children():
            values.append(-negamax(child, depth - 1, evaluation_function))
        # print(f'values={values}')
        value = max(values)
    # print(f'depth={depth} player={node.player.id} value={value:3}')
    return value

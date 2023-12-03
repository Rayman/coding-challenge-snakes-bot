import os.path
from dataclasses import dataclass
from functools import partial
from typing import Tuple

import torch

from .board import Node
from .evaluation_functions import prefer_eating
from .parse_matches import Record
from .search_functions import negamax_ab_move
from .snake import FastSnake
from .train import record_to_observation, NeuralNetwork
from ...bot import Bot
from ...constants import Move

__all__ = ['Slifer']


@dataclass
class Parameters:
    battle_mode_stop: int = 5  # length margin lower than this will enable eating mode
    battle_mode_margin: int = 5  # battle mode starts at battle_mode_stop + battle_mode_margin

    eating_mode_stop: int = 8  # length when to stop eating mode and switch to eating_and_battle
    # candy_voronoi_scaling: float = 0.18587348196111586  # Prefer candy (0) vs voronoi (1)
    candy_voronoi_scaling: float = 0.9  # Prefer candy (0) vs voronoi (1)
    voronoi_max: float = 25
    # length_scaling: float = 0.5460890971075978
    length_scaling: float = 0.3


def nn_evaluation(node: Node, model):
    record = Record(player=node.player, opponent=node.opponent, candies=node.candies, winner=1)
    observation = record_to_observation(record)
    pred = model(observation).item()
    return pred


class Sudowoodo(Bot):
    def __init__(self, id: int, grid_size: Tuple[int, int], depth=2):
        super().__init__(id=id, grid_size=grid_size)
        self.battle_mode = False
        self.depth = depth
        self.parameters = Parameters()
        self.model = NeuralNetwork(512)
        self.model.load_state_dict(torch.load(os.path.join(os.path.dirname(__file__), 'models', 'model_10.pth')))

    @property
    def name(self):
        return 'Sudowoodo'

    @property
    def contributor(self):
        return 'Rayman'

    def determine_next_move(self, snake, other_snakes, candies) -> Move:
        player = FastSnake(id=snake.id, positions=snake.positions)
        opponent = FastSnake(id=other_snakes[0].id, positions=other_snakes[0].positions)

        length_margin = len(player) * 2 - len(opponent)
        # closest_candy = min(np.linalg.norm(snake[0] - c, 1) for c in candies)
        # player_distance = abs(player[0][0] - opponent[0][0]) + abs(player[0][1] - opponent[0][1])

        if length_margin > self.parameters.battle_mode_stop + self.parameters.battle_mode_margin:
            self.battle_mode = True
        elif length_margin < self.parameters.battle_mode_stop:
            self.battle_mode = False

        if self.battle_mode:
            return negamax_ab_move(self.grid_size, player, opponent, candies, self.depth,
                                   partial(nn_evaluation, model=self.model))
        else:
            return negamax_ab_move(self.grid_size, player, opponent, candies, self.depth, prefer_eating)


class Sudowoodo0(Sudowoodo):
    def __init__(self, id: int, grid_size: Tuple[int, int]):
        super().__init__(id=id, grid_size=grid_size, depth=0)

    @property
    def name(self):
        return 'Sudowoodo0'


class Sudowoodo1(Sudowoodo):
    def __init__(self, id: int, grid_size: Tuple[int, int]):
        super().__init__(id=id, grid_size=grid_size, depth=1)

    @property
    def name(self):
        return 'Sudowoodo1'


class Sudowoodo2(Sudowoodo):
    def __init__(self, id: int, grid_size: Tuple[int, int]):
        super().__init__(id=id, grid_size=grid_size, depth=2)

    @property
    def name(self):
        return 'Sudowoodo2'


class Sudowoodo3(Sudowoodo):
    def __init__(self, id: int, grid_size: Tuple[int, int]):
        super().__init__(id=id, grid_size=grid_size, depth=3)

    @property
    def name(self):
        return 'Sudowoodo3'

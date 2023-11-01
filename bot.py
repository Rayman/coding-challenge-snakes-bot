import json
import os.path
from dataclasses import dataclass
from functools import partial
from typing import Tuple

from .evaluation_functions import prefer_eating, prefer_battle, eat_and_battle, snek_evaluate
from .search_functions import negamax_ab_move
from .snake import FastSnake
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


class Slifer(Bot):
    def __init__(self, id: int, grid_size: Tuple[int, int], depth=2):
        super().__init__(id=id, grid_size=grid_size)
        self.battle_mode = False
        self.depth = depth
        self.parameters = Parameters()

    @property
    def name(self):
        return 'Slifer the Sky Dragon'

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
            return negamax_ab_move(self.grid_size, player, opponent, candies, self.depth, prefer_battle)
        else:
            return negamax_ab_move(self.grid_size, player, opponent, candies, self.depth, prefer_eating)


class Hybrid(Bot):
    def __init__(self, id: int, grid_size: Tuple[int, int], depth=0):
        super().__init__(id=id, grid_size=grid_size)
        self.battle_mode = False
        self.depth = depth
        self.parameters = Parameters()

    @property
    def name(self):
        return 'Hybrid'

    @property
    def contributor(self):
        return 'Rayman'

    def determine_next_move(self, snake, other_snakes, candies) -> Move:
        player = FastSnake(id=snake.id, positions=snake.positions)
        opponent = FastSnake(id=other_snakes[0].id, positions=other_snakes[0].positions)

        if len(player) >= self.parameters.eating_mode_stop:
            return negamax_ab_move(self.grid_size, player, opponent, candies, self.depth,
                                   partial(eat_and_battle, parameters=self.parameters))
        else:
            return negamax_ab_move(self.grid_size, player, opponent, candies, self.depth, prefer_eating)


class SnekClone(Bot):
    def __init__(self, id: int, grid_size: Tuple[int, int], depth=0):
        super().__init__(id=id, grid_size=grid_size)
        self.depth = depth
        with open(os.path.join(os.path.dirname(__file__), 'territory.json')) as f:
            self.territory = json.load(f)

    @property
    def name(self):
        return 'SnekClone'

    @property
    def contributor(self):
        return 'Rayman'

    def determine_next_move(self, snake, other_snakes, candies) -> Move:
        player = FastSnake(id=snake.id, positions=snake.positions)
        opponent = FastSnake(id=other_snakes[0].id, positions=other_snakes[0].positions)
        return negamax_ab_move(self.grid_size, player, opponent, candies, self.depth,
                               partial(snek_evaluate, territory=self.territory))


class Hybrid1(Slifer):
    def __init__(self, id: int, grid_size: Tuple[int, int]):
        super().__init__(id=id, grid_size=grid_size, depth=1)

    @property
    def name(self):
        return 'Hybrid1'


class Hybrid2(Slifer):
    def __init__(self, id: int, grid_size: Tuple[int, int]):
        super().__init__(id=id, grid_size=grid_size, depth=2)

    @property
    def name(self):
        return 'Hybrid2'


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


class Slifer3(Slifer):
    def __init__(self, id: int, grid_size: Tuple[int, int]):
        super().__init__(id=id, grid_size=grid_size, depth=3)

    @property
    def name(self):
        return 'Slifer3'

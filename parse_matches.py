#!/usr/bin/env python3
import pickle
from argparse import ArgumentParser, FileType
from copy import deepcopy
from dataclasses import dataclass
from random import randint
from typing import List

import numpy as np
import yaml
from tqdm import tqdm

from ...game import RoundType
from ...replay import ReplayReader
from ...snake import Snake


@dataclass
class Record:
    player: Snake
    opponent: Snake
    candies: List[np.array]
    winner: int


def main(match):
    records = load_records(match)
    with open(f'{match.name}.pkl', 'wb') as f:
        pickle.dump(records, f)


def load_records(match):
    docs = tqdm(yaml.safe_load_all(match))
    docs.set_description('Loading matches')
    docs = list(docs)

    records = []
    docs = tqdm(docs)
    docs.set_description('Processing matches')
    sample = 2
    for doc in docs:
        for state in ReplayReader(doc).states():
            player_rank = doc['rank'][state.turn]
            opponent_rank = doc['rank'][(state.turn + 1) % 2]
            winner = 1 if player_rank < opponent_rank else -1 if player_rank > opponent_rank else 0
            if sample > 1:
                # Downsample, using only 1/Nth of the items
                if randint(0, sample - 1) != 0:
                    continue  # Skip this record
                assert state.round_type == RoundType.TURNS
                assert len(state.snakes) == 2
                player = state.snakes[state.turn]
                opponent = state.snakes[(state.turn + 1) % 2]
                if player is None or opponent is None:
                    continue
                records.append(
                    Record(player=deepcopy(player), opponent=deepcopy(opponent), candies=deepcopy(state.candies),
                           winner=winner))
    return records


if __name__ == '__main__':
    parser = ArgumentParser(description='Replay a match')
    parser.add_argument('match', type=FileType('r'), help="Input match database")
    args = parser.parse_args()

    try:
        main(**vars(args))
    except KeyboardInterrupt:
        pass

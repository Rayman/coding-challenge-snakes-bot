from argparse import ArgumentParser, FileType

import matplotlib.pyplot as plt
import torch
import yaml

from .parse_matches import Record
from .train import NeuralNetwork, record_to_observation
from ...game import RoundType
from ...replay import ReplayReader


def main(match, network):
    model = NeuralNetwork(512)
    model.load_state_dict(torch.load(network))
    model.eval()

    docs = yaml.safe_load_all(match)
    with torch.no_grad():
        for doc in docs:
            predictions = [[], []]
            for state in ReplayReader(doc).states():
                player_rank = doc['rank'][state.turn]
                opponent_rank = doc['rank'][(state.turn + 1) % 2]
                winner = 1 if player_rank < opponent_rank else -1 if player_rank > opponent_rank else 0
                assert state.round_type == RoundType.TURNS
                assert len(state.snakes) == 2
                player = state.snakes[state.turn]
                opponent = state.snakes[(state.turn + 1) % 2]
                if player is None or opponent is None:
                    continue

                record = Record(player=player, opponent=opponent, candies=state.candies, winner=winner)
                observation = record_to_observation(record)
                pred = model(observation).item()
                predictions[state.turn].append(pred)
            plt.plot(predictions[0])
            plt.plot(predictions[1])
            break
    plt.legend()
    plt.show()


if __name__ == '__main__':
    parser = ArgumentParser(description='Replay a match')
    parser.add_argument('match', type=FileType('r'), help="Input match database")
    parser.add_argument('network', type=FileType('rb'), help="Neural Network")
    args = parser.parse_args()

    try:
        main(**vars(args))
    except KeyboardInterrupt:
        pass

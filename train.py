#!/usr/bin/env python3
from argparse import ArgumentParser, FileType
from copy import deepcopy
from dataclasses import dataclass
from random import randint
from typing import List, Dict

import numpy as np
import torch
import yaml
from torch import nn, Tensor
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
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

    dataset = []
    for record in records:
        dataset.append((record_to_observation(record), Tensor([record.winner])))

    train_dataset, test_dataset = random_split(dataset, (0.9, 0.1))

    batch_size = 64
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size)

    for X, y in train_dataloader:
        print(f"Shape of X: {X.shape} {X.dtype}")
        print(f"Shape of y: {y.shape} {y.dtype}")
        break

    model = NeuralNetwork(512)

    loss_fn = nn.MSELoss()
    learning_rate = 1e-3
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    epochs = 10
    writer = SummaryWriter()
    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}\n-------------------------------")
        train_loop(train_dataloader, model, loss_fn, optimizer, writer, epoch)
        test_loop(test_dataloader, model, loss_fn, writer, epoch)
    print('Done!')


def record_to_observation(record: Record):
    player = torch.zeros((16, 16))
    for segment in record.player:
        player[tuple(segment)] = 1
    opponent = torch.zeros((16, 16))
    for segment in record.opponent:
        opponent[tuple(segment)] = 1

    # return Tensor(np.concatenate((record.player[0], record.opponent[0])))
    return Tensor(np.concatenate((player.flatten(), opponent.flatten())))


def train_loop(dataloader, model, loss_fn, optimizer, writer, epoch):
    size = len(dataloader.dataset)
    # Set the model to training mode - important for batch normalization and dropout layers
    # Unnecessary in this situation but added for best practices
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        # Compute prediction and loss
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 100 == 99:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

            tb_x = (epoch * len(dataloader) + batch + 1) * dataloader.batch_size
            writer.add_scalar('Loss/train', loss, tb_x)


def test_loop(dataloader, model, loss_fn, writer, epoch):
    # Set the model to evaluation mode - important for batch normalization and dropout layers
    # Unnecessary in this situation but added for best practices
    model.eval()
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    # Evaluating the model with torch.no_grad() ensures that no gradients are computed during test mode
    # also serves to reduce unnecessary gradient computations and memory usage for tensors with requires_grad=True
    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (torch.round(pred) == y).sum().item()

    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100 * correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

    tb_x = (epoch * len(dataloader) + 1) * dataloader.batch_size
    print(f'len={len(dataloader)} shape={X.shape} i={tb_x}')
    writer.add_scalar('Loss/test', test_loss, tb_x)
    writer.add_scalar('Accuracy/test', 100 * correct, tb_x)


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


class NeuralNetwork(nn.Module):
    def __init__(self, feature_dim: int):
        super().__init__()
        self.linear_relu = nn.Sequential(
            nn.Linear(feature_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Tanh(),
        )

    def forward(self, observations: Dict[str, torch.Tensor]):
        return self.linear_relu(observations)


if __name__ == '__main__':
    parser = ArgumentParser(description='Replay a match')
    parser.add_argument('match', type=FileType('r'), help="Input match database")
    args = parser.parse_args()

    try:
        main(**vars(args))
    except KeyboardInterrupt:
        pass

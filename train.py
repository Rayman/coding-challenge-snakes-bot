#!/usr/bin/env python3
import pickle
import sys
from argparse import ArgumentParser, FileType
from typing import Dict

import numpy as np
import torch
from torch import nn, Tensor
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter

from .parse_matches import Record


def main(match):
    records = pickle.load(match)
    dataset = []
    for record in records:
        dataset.append((record_to_observation(record), Tensor([record.winner])))
    del records

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

    epochs = 100
    writer = SummaryWriter()
    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}\n-------------------------------")
        train_loop(train_dataloader, model, loss_fn, optimizer, writer, epoch)
        test_loop(test_dataloader, model, loss_fn, writer, epoch)
        torch.save(model.state_dict(), f'model_{epoch}.pth')
    print('Done!')


def record_to_observation(record: Record):
    player = torch.zeros((16, 16))
    opponent = torch.zeros((16, 16))
    player[tuple(np.array(record.player.positions).T)] = 1
    opponent[tuple(np.array(record.opponent.positions).T)] = 1

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


def get_size(obj, seen=None):
    """Recursively finds size of objects"""
    size = sys.getsizeof(obj)
    if seen is None:
        seen = set()
    obj_id = id(obj)
    if obj_id in seen:
        return 0
    # Important mark as seen *before* entering recursion to gracefully handle
    # self-referential objects
    seen.add(obj_id)
    if isinstance(obj, dict):
        size += sum([get_size(v, seen) for v in obj.values()])
        size += sum([get_size(k, seen) for k in obj.keys()])
    elif hasattr(obj, '__dict__'):
        size += get_size(obj.__dict__, seen)
    elif hasattr(obj, '__iter__') and not isinstance(obj, (str, bytes, bytearray)):
        size += sum([get_size(i, seen) for i in obj])
    return size


if __name__ == '__main__':
    parser = ArgumentParser(description='Replay a match')
    parser.add_argument('match', type=FileType('rb'), help="Input match database")
    args = parser.parse_args()

    try:
        main(**vars(args))
    except KeyboardInterrupt:
        pass

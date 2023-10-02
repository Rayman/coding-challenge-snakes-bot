from collections.abc import Sequence

import numpy as np

from ...constants import RIGHT, UP
from ...snake import Snake


class FastSnake(Sequence):
    def __init__(self, id, positions):
        assert len(positions.shape) == 2
        assert positions.shape[1] == 2
        self.id = id
        self.positions = [(int(x), int(y)) for x, y in positions]

    def move(self, move, grow=False):
        head = self.positions[0]
        head = (head[0] + move[0], head[1] + move[1])
        if grow:
            self.positions = [head] + self.positions
        else:
            self.positions[1:] = self.positions[0:-1]
            self.positions[0] = head

    def __iter__(self):
        return iter(self.positions)

    def __len__(self):
        return len(self.positions)

    def __getitem__(self, i):
        return self.positions[i]

    def __repr__(self):
        return f'id={self.id} positions={self.positions}'

    def collides(self, pos):
        return pos in self.positions


def test_benchmark_move(benchmark):
    length = 50
    snake = Snake(id=0, positions=np.vstack(([1, 3], np.tile([0, 3], (length, 1)))))
    benchmark(snake.move, RIGHT)
    print(snake)


def test_benchmark_move_fast(benchmark):
    length = 50
    snake = FastSnake(id=0, positions=np.vstack(([1, 3], np.tile([0, 3], (length, 1)))))
    benchmark(snake.move, RIGHT)
    print(snake)


def test_benchmark_grow(benchmark):
    length = 50

    @benchmark
    def grow():
        snake = Snake(id=0, positions=np.vstack(([1, 3], np.tile([0, 3], (2, 1)))))
        for _ in range(length):
            snake.move(RIGHT, grow=True)


def test_benchmark_grow_fast(benchmark):
    length = 50

    @benchmark
    def grow():
        snake = FastSnake(id=0, positions=np.vstack(([1, 3], np.tile([0, 3], (2, 1)))))
        for _ in range(length):
            snake.move(RIGHT, grow=True)


def test_benchmark_collides(benchmark):
    length = 50
    snake = Snake(id=0, positions=np.vstack(([1, 3], np.tile([0, 3], (length, 1)))))
    assert not benchmark(snake.collides, (-1, -1))


def test_benchmark_collides_fast(benchmark):
    length = 50
    snake = FastSnake(id=0, positions=np.vstack(([1, 3], np.tile([0, 3], (length, 1)))))
    assert not benchmark(snake.collides, (-1, -1))


def test_snake_sequence():
    head = np.array([5, 6])
    snake = FastSnake(0, np.array([head, head + UP, head + UP + RIGHT]))
    assert len(snake) == 3
    assert np.array_equal(snake[0], [5, 6])
    assert np.array_equal(snake[1], [5, 7])
    assert np.array_equal(snake[2], [6, 7])


def test_snake_grow():
    head = np.array([5, 6])
    snake = FastSnake(0, np.array([head]))
    snake.move(UP)
    assert len(snake) == 1
    assert np.array_equal(snake[0], [5, 7])

    snake.move(UP, grow=True)
    assert len(snake) == 2
    assert np.array_equal(snake[0], [5, 8])
    assert np.array_equal(snake[1], [5, 7])

    snake.move(UP)
    assert len(snake) == 2
    assert np.array_equal(snake[0], [5, 9])
    assert np.array_equal(snake[1], [5, 8])


def test_snake_collide():
    head = np.array([5, 6])
    snake = FastSnake(0, np.array([[5, 6], [5, 7], [6, 7]]))
    assert snake.collides((5, 6))
    assert snake.collides((5, 7))
    assert snake.collides((6, 7))
    assert not snake.collides((6, 6))

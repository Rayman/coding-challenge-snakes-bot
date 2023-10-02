import numpy as np

from .snake import FastSnake
from ...constants import RIGHT, UP
from ...snake import Snake


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

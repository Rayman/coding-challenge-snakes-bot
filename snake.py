from collections.abc import Sequence


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

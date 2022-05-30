import math

I = [
    [0, 0, 0, 0],
    [1, 1, 1, 1],
    [0, 0, 0, 0],
    [0, 0, 0, 0]
]

O = [
    [2, 2],
    [2, 2]
]

T = [
    [0, 3, 0],
    [3, 3, 3],
    [0, 0, 0]
]

S = [
    [0, 4, 4],
    [4, 4, 0],
    [0, 0, 0]
]

Z = [
    [5, 5, 0],
    [0, 5, 5],
    [0, 0, 0]
]

J = [
    [6, 0, 0],
    [6, 6, 6],
    [0, 0, 0]
]

L = [
    [0, 0, 7],
    [7, 7, 7],
    [0, 0, 0]
]

ALL_FIGURES = [I, O, T, S, Z, J, L]


class Figure:

    def __init__(self, x, y, shape):
        self.shape = shape

        self.height = len(self.shape)
        self.width = len(self.shape[0])

        self.x = x - math.ceil(self.width / 2)
        self.y = y + self.height

        self.color = max([max(row) for row in self.shape])

    def rotate_clockwise(self):
        self.shape = [[self.shape[y][x] for y in range(len(self.shape))] for x in range(len(self.shape[0]) - 1, -1, -1)]
        self.height = len(self.shape)
        self.width = len(self.shape[0])

    def rotate_anticlockwise(self):
        self.shape = [[self.shape[y][x] for y in range(len(self.shape) - 1, -1, -1)] for x in range(len(self.shape[0]))]
        self.height = len(self.shape)
        self.width = len(self.shape[0])

    def move_left(self):
        self.x -= 1

    def move_right(self):
        self.x += 1

    def move_down(self):
        self.y += 1

    def move_up(self):
        self.y -= 1

    def is_occupied(self, x, y):
        return self.shape[y][x] > 0


import math


I = [
    [[1, 1, 1, 1]],
    [[1], [1], [1], [1]]
]

O = [
    [[2, 2], [2, 2]]
]

T = [
    [[0, 3, 0], [3, 3, 3]],
    [[3, 0], [3, 3], [3, 0]],
    [[3, 3, 3], [0, 3, 0]],
    [[0, 3], [3, 3], [0, 3]]
]

S = [
    [[0, 4, 4], [4, 4, 0]],
    [[4, 0], [4, 4], [0, 4]]
]

Z = [
    [[5, 5, 0], [0, 5, 5]],
    [[0, 5], [5, 5], [5, 0]]
]

J = [
    [[6, 0, 0], [6, 6, 6]],
    [[6, 6], [6, 0], [6, 0]],
    [[6, 6, 6], [0, 0, 6]],
    [[0, 6], [0, 6], [6, 6]]
]

L = [
    [[0, 0, 7], [7, 7, 7]],
    [[7, 0], [7, 0], [7, 7]],
    [[7, 7, 7], [7, 0, 0]],
    [[7, 7], [0, 7], [0, 7]]
]

ALL_FIGURES = [I, O, T, S, Z, J, L]


class Figure:

    def __init__(self, x, y, shapes):
        self.shapes = shapes

        self.max_rotation_index = len(shapes)
        self.current_rotation_index = 0

        self.shape = self.shapes[self.current_rotation_index]

        self.height = len(self.shape)
        self.width = len(self.shape[0])

        self.x = x - math.ceil(self.width / 2)
        self.y = y

        self.color = max([max(row) for row in self.shape])

    def set_rotation_index(self, rotation_index):
        self.current_rotation_index = rotation_index % self.max_rotation_index
        self.shape = self.shapes[self.current_rotation_index]
        self.height = len(self.shape)
        self.width = len(self.shape[0])

    def rotate_clockwise(self):
        self.set_rotation_index(self.current_rotation_index + 1)

    def rotate_anticlockwise(self):
        self.set_rotation_index(self.current_rotation_index - 1)

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


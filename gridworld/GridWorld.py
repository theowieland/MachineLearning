class Coordinate:

    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __add__(self, other):
        return Coordinate(self.x + other.x, self.y + other.y)

    def __sub__(self, other):
        return Coordinate(self.x - other.x, self.y - other.y)

    def __str__(self):
        return "(" + str(self.x) + ", " + str(self.y) + ")"


ACTION_UP = Coordinate(0, 1)
ACTION_DOWN = Coordinate(0, -1)
ACTION_LEFT = Coordinate(-1, 0)
ACTION_RIGHT = Coordinate(1, 0)


class GridWorld:

    def __init__(self, width, height):
        self.width = width
        self.height = height

        self.grid = [[0 for _x in range(width)] for _y in range(height)]

    def set(self, coordinate, state):
        if self.is_inside(coordinate):
            self.grid[coordinate.y][coordinate.x] = state

    def get(self, coordinate):
        return self.grid[coordinate.y][coordinate.x]

    def fill(self, state):
        for x in range(self.width):
            for y in range(self.height):
                self.set(Coordinate(x, y), state)

    def is_inside(self, coordinate, border=0):
        return border <= coordinate.x < self.width - border and border <= coordinate.y < self.height - border


import random

from games.tetris import Figures
from games.tetris.Figures import Figure

OCCUPIED = 1
EMPTY = 0

STATE_RUNNING = 0
STATE_GAME_OVER = 1


# (0, 0) is the top left coordinate
class Tetris:

    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.board = [[0 for _x in range(self.width)] for _y in range(self.height)]
        self.current_figure = None
        self.current_state = STATE_RUNNING
        self.score = 0

        self.reset()

    def reset(self):
        self.score = 0
        self.current_state = STATE_RUNNING
        self.board = [[0 for _x in range(self.width)] for _y in range(self.height)]
        self.choose_random_figure()

    def choose_random_figure(self):
        random_shape = Figures.ALL_FIGURES[random.randint(0, len(Figures.ALL_FIGURES) - 1)]
        self.current_figure = Figure(int(self.width / 2), 0, random_shape)

        if self.intersects():
            self.current_state = STATE_GAME_OVER
            self.current_figure = None

    def remove_full_rows(self):
        for row in range(self.height - 1, -1, -1):
            if all([self.board[row][x] > 0 for x in range(self.width)]):
                self.score += 1

                # move all rows above down
                for row_above in range(row - 1, 0, -1):
                    for x in range(self.width):
                        self.board[row_above + 1][x] = self.board[row_above][x]

    def intersects(self):
        for figure_y in range(self.current_figure.height):
            for figure_x in range(self.current_figure.width):
                if self.current_figure.is_occupied(figure_x, figure_y):
                    if self.current_figure.y + figure_y >= self.height \
                            or self.current_figure.x + figure_x < 0 \
                            or self.current_figure.x + figure_x >= self.width:
                        return True

                    if self.is_occupied(self.current_figure.x + figure_x, self.current_figure.y + figure_y):
                        return True

        return False

    def move_down(self):
        self.current_figure.move_down()
        if self.intersects():
            self.current_figure.move_up()

            self.place_current_figure()

    def move_left(self):
        self.current_figure.move_left()
        if self.intersects():
            self.current_figure.move_right()

    def move_right(self):
        self.current_figure.move_right()
        if self.intersects():
            self.current_figure.move_left()

    def place_current_figure(self):
        figure = self.current_figure

        for figure_y in range(figure.height):
            for figure_x in range(figure.width):
                if figure.is_occupied(figure_x, figure_y):
                    self.board[figure.y + figure_y][figure.x + figure_x] = figure.color

        self.remove_full_rows()
        self.choose_random_figure()

    def rotate_current_figure(self):
        self.current_figure.rotate_clockwise()
        if self.intersects():
            self.current_figure.rotate_anticlockwise()

    def is_occupied(self, x, y):
        return self.board[y][x] > 0

    def get_state(self, x, y):
        return self.board[y][x]

    def get_game_state(self):
        return self.current_state

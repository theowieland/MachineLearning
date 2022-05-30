import random
from copy import deepcopy

import numpy as np

from games.tetris import Figures
from games.tetris.Figures import Figure

OCCUPIED = 1
EMPTY = 0

ACTION_MOVE_DOWN = 0
ACTION_MOVE_LEFT = 1
ACTION_MOVE_RIGHT = 2
ACTION_ROTATE_CLOCKWISE = 3

BOARD_HEIGHT = 15
BOARD_WIDTH = 7


# (0, 0) is the top left coordinate
class Tetris:

    @staticmethod
    def create_empty_board():
        return np.zeros((BOARD_HEIGHT, BOARD_WIDTH), dtype=int)

    @staticmethod
    def clear_full_rows(board):
        num_cleared_rows = 0

        row = len(board) - 1
        while row >= 0:
            if all([board[row][x] > EMPTY for x in range(BOARD_WIDTH)]):
                num_cleared_rows += 1

                # move all rows above down
                for row_above in range(row - 1, 0, -1):
                    for x in range(BOARD_WIDTH):
                        board[row_above + 1][x] = board[row_above][x]
            else:
                row -= 1

        return num_cleared_rows

    @staticmethod
    def is_inside_board(x, y):
        return 0 <= x < BOARD_WIDTH and 0 <= y < BOARD_HEIGHT

    @staticmethod
    def get_random_figure():
        random_shape = Figures.ALL_FIGURES[random.randint(0, len(Figures.ALL_FIGURES) - 1)]

        return Figure(int(BOARD_WIDTH / 2), 0, random_shape)

    @staticmethod
    def intersects(board, figure):
        for figure_y in range(figure.height):
            for figure_x in range(figure.width):
                if figure.is_occupied(figure_x, figure_y):
                    if not Tetris.is_inside_board(figure_x + figure.x, figure_y + figure.y):
                        return True

                    if Tetris.is_occupied(board, figure.x + figure_x, figure.y + figure_y):
                        return True

        return False

    @staticmethod
    def place_figure(board, figure):
        for figure_y in range(figure.height):
            for figure_x in range(figure.width):
                if figure.is_occupied(figure_x, figure_y):
                    if Tetris.is_inside_board(figure_x + figure.x, figure_y + figure.y):
                        board[figure_y + figure.y][figure_x + figure.x] = figure.color

    @staticmethod
    def drop_figure(board, figure):
        if Tetris.intersects(board, figure):
            return False

        while not Tetris.intersects(board, figure):
            figure.move_down()

        figure.move_up()
        Tetris.place_figure(board, figure)
        return True

    @staticmethod
    def move_figure_down(board, figure: Figure):
        figure.move_down()

        if Tetris.intersects(board, figure):
            figure.move_up()

    @staticmethod
    def move_figure_right(board, figure: Figure):
        figure.move_right()

        if Tetris.intersects(board, figure):
            figure.move_left()

    @staticmethod
    def move_figure_left(board, figure: Figure):
        figure.move_left()

        if Tetris.intersects(board, figure):
            figure.move_right()

    @staticmethod
    def rotate_figure(board, figure):
        figure.rotate_clockwise()

        if Tetris.intersects(board, figure):
            figure.rotate_anticlockwise()

    @staticmethod
    def simulate_all_boards(board, figure):
        states = list()

        for column in range(BOARD_WIDTH):
            for rotation in range(0, 4):
                board_copy = deepcopy(board)
                figure_copy = deepcopy(figure)

                for _rotate in range(0, rotation):
                    figure_copy.rotate_clockwise()

                figure_copy.x = column

                if not Tetris.intersects(board_copy, figure_copy):
                    Tetris.drop_figure(board_copy, figure_copy)
                    reward = Tetris.clear_full_rows(board_copy) + (Tetris.get_average_column_height(board) - Tetris.get_average_column_height(board_copy))

                    states.append((column, rotation, reward, board_copy))

        return states

    @staticmethod
    def simulate_action(board, figure, column, rotation):
        board_copy = deepcopy(board)
        figure_copy = deepcopy(figure)

        for _rotate in range(0, rotation):
            figure_copy.rotate_clockwise()

        figure_copy.x = column
        Tetris.drop_figure(board_copy, figure_copy)

        reward = Tetris.clear_full_rows(board_copy) + (Tetris.get_average_column_height(board) - Tetris.get_average_column_height(board_copy))

        return board_copy, reward

    @staticmethod
    def get_average_column_height(board):
        column_height = [0] * BOARD_WIDTH

        for column in range(0, BOARD_WIDTH):
            y = 0

            while not Tetris.is_occupied(board, column, y) and y < BOARD_HEIGHT - 1:
                y += 1

            column_height[column] = BOARD_HEIGHT - y

        return np.average(column_height)

    @staticmethod
    def is_occupied(board, x, y):
        return board[y][x] != EMPTY

    @staticmethod
    def feature_vector(board, figure):
        heights = BOARD_HEIGHT - (board != EMPTY).argmax(axis=0)
        heights = np.array([0 if height == BOARD_HEIGHT else height for height in heights])

        bumpiness = np.ediff1d(heights)
        sum_height_diff = sum(np.abs(bumpiness))

        sum_height = sum(heights)
        max_height = max(heights)
        average_height = np.average(heights)

        holes = np.zeros(BOARD_WIDTH, dtype=int)

        for x in range(0, BOARD_WIDTH):
            num_holes = 0
            for y in range(BOARD_HEIGHT - 1, 0, -1):
                if not Tetris.is_occupied(board, x, y) and Tetris.is_occupied(board, x, y - 1):
                    num_holes += 1

            holes[x] = num_holes

        sum_holes = sum(holes)

        return np.concatenate(([0], [max_height, sum_height, sum_holes, average_height, sum_height_diff, 1]), axis=0)

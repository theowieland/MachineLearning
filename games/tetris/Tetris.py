import random
from copy import deepcopy

import numpy as np

from games.tetris import Figures
from games.tetris.Figures import Figure

OCCUPIED = 1
EMPTY = 0

BOARD_HEIGHT = 20
BOARD_WIDTH = 12


# (0, 0) is the top left coordinate
class Tetris:

    @staticmethod
    def create_empty_board():
        return np.zeros((BOARD_HEIGHT, BOARD_WIDTH), dtype=int)

    @staticmethod
    def clear_full_rows(board):
        # detect full rows
        full_rows = list()
        for row in range(BOARD_HEIGHT):
            if all(board[row][x] > EMPTY for x in range(BOARD_WIDTH)):
                full_rows.append(row)

        num_cleared_rows = len(full_rows)

        # remove full rows
        if len(full_rows) > 0:  # there are lines to clear
            row_to_copy_to = full_rows[len(full_rows) - 1]

            for index in range(len(full_rows) - 1, 0, -1):
                for partial_row in range(full_rows[index] - 1, full_rows[index - 1], -1):
                    for x in range(BOARD_WIDTH):
                        board[row_to_copy_to][x] = board[partial_row][x]

                    row_to_copy_to -= 1

            for remaining_row in range(full_rows[0] - 1, len(full_rows), -1):
                for x in range(BOARD_WIDTH):
                    board[row_to_copy_to][x] = board[remaining_row][x]

                row_to_copy_to -= 1

            for row_to_empty in range(0, len(full_rows)):
                for x in range(BOARD_WIDTH):
                    board[row_to_empty][x] = EMPTY

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

        current_board_height_sum = sum(Tetris.get_column_heights(board))

        for column in range(-figure.width, BOARD_WIDTH + figure.width - 1):
            for rotation in range(0, figure.max_rotation_index):
                board_copy = deepcopy(board)
                figure_copy = deepcopy(figure)

                figure_copy.set_rotation_index(rotation)

                figure_copy.x = column
                figure_copy.y = 0

                if not Tetris.intersects(board_copy, figure_copy):
                    Tetris.drop_figure(board_copy, figure_copy)

                    num_cleared_rows = Tetris.clear_full_rows(board_copy)
                    feature_vector = Tetris.feature_vector(board_copy)

                    next_figure = Tetris.get_random_figure()
                    if Tetris.intersects(board_copy, next_figure):
                        reward = -50
                    else:
                        reward = num_cleared_rows + ((current_board_height_sum - sum(Tetris.get_column_heights(board_copy))))

                    states.append((column, rotation, reward, board_copy, next_figure, feature_vector))

        return states

    @staticmethod
    def get_column_heights(board):
        heights = BOARD_HEIGHT - (board != EMPTY).argmax(axis=0)
        heights = np.array([0 if height == BOARD_HEIGHT else height for height in heights])

        return heights

    @staticmethod
    def get_average_column_height(board):
        return np.average(Tetris.get_column_heights(board))

    @staticmethod
    def is_occupied(board, x, y):
        return board[y][x] != EMPTY

    @staticmethod
    def feature_vector(board):
        heights = Tetris.get_column_heights(board)

        bumpiness = np.ediff1d(heights)
        sum_height_diff = sum(np.abs(bumpiness))

        sum_height = sum(heights)
        max_height = max(heights)
        average_height = np.average(heights)

        holes = np.zeros(BOARD_WIDTH, dtype=int)

        for x in range(0, BOARD_WIDTH):
            num_holes = 0
            occupied_above = False
            for y in range(0, BOARD_HEIGHT):
                if not occupied_above and Tetris.is_occupied(board, x, y):
                    occupied_above = True
                elif occupied_above and not Tetris.is_occupied(board, x, y):
                    num_holes += 1

            holes[x] = num_holes

        sum_holes = sum(holes)

        return np.concatenate((
            [sum_height, average_height, sum_height_diff, sum_holes, max_height],
            [1]
        ), axis=0)

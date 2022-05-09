from abc import ABC

from gridworld.GridWorld import GridWorld, Coordinate
import random


class GridWorldGenerator(ABC):

    def __init__(self, world: GridWorld):
        self.world = world

    def create_border(self, border_state, thickness):
        for border_offset in range(thickness):
            for y in range(self.world.height):
                self.world.set(Coordinate(border_offset, y), border_state)
                self.world.set(Coordinate(self.world.width - 1 - border_offset, y), border_state)

            for x in range(self.world.width):
                self.world.set(Coordinate(x, border_offset), border_state)
                self.world.set(Coordinate(x, self.world.height - 1 - border_offset), border_state)

    def generate(self):
        pass


class PrimsMazeGenerator(GridWorldGenerator):

    def __init__(self, world, wall_state, passage_state):
        self.wall_state = wall_state
        self.passage_state = passage_state

        super().__init__(world)

    def generate(self):
        border_thickness = 2

        self.create_border(self.wall_state, thickness=2)
        self.world.fill(self.wall_state)

        odd_start = border_thickness + (1 if border_thickness % 2 == 0 else 0)
        odd_stop_x = self.world.width - 1 - border_thickness - (1 if (self.world.width - 1 - border_thickness) % 2 == 0 else 0)
        odd_stop_y = self.world.height - 1 - border_thickness - (1 if (self.world.height - 1 - border_thickness) % 2 == 0 else 0)
        random_x = random.randrange(odd_start, odd_stop_x + 1, 2)  # create odd number in range
        random_y = random.randrange(odd_start, odd_stop_y + 1, 2)  # create odd number in range

        start = Coordinate(random_x, random_y)
        self.world.set(start, self.passage_state)

        frontiers = list()

        for passage, frontier in self.get_frontiers(start):
            if self.world.is_inside(frontier, border_thickness):
                frontiers.append((passage, frontier))

        while len(frontiers) > 0:
            random_frontier_index = random.randint(0, len(frontiers) - 1)
            random_passage, random_frontier = frontiers[random_frontier_index]
            frontiers.pop(random_frontier_index)  # remove selected frontier from frontiers

            if self.world.get(random_frontier) == self.wall_state:
                self.world.set(random_frontier, self.passage_state)
                self.world.set(random_passage, self.passage_state)

            for target_passage, target_coordinate in self.get_frontiers(random_frontier):
                if self.world.is_inside(target_coordinate, border_thickness) and self.world.get(target_coordinate) == self.wall_state and self.world.get(target_passage) == self.wall_state:
                    frontiers.append((target_passage, target_coordinate))

    def get_frontiers(self, cell: Coordinate):
        return [(cell + direction, cell + 2 * direction) for direction in
                [Coordinate(-1, 0), Coordinate(1, 0), Coordinate(0, -1), Coordinate(0, 1)]]


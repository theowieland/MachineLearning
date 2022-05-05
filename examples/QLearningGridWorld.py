import random

from gridworld import GridWorldDrawer
from gridworld.GridWorld import GridWorld, Coordinate
from gridworld.GridWorldDrawer import draw_world, draw_shape
from qlearning.TabularQLearning import TabularQLearning

ACTION_UP = Coordinate(0, 1)
ACTION_DOWN = Coordinate(0, -1)
ACTION_LEFT = Coordinate(-1, 0)
ACTION_RIGHT = Coordinate(1, 0)

WALL_TILE = 0
EMPTY_TILE = 1
GOAL_TILE = 2


class QLearningGridWorld(TabularQLearning):

    def __init__(self, world: GridWorld, goal_locations):
        super(QLearningGridWorld, self).__init__(world.width * world.height, 4, discount_factor=.98, learning_rate=0.1)

        self.world = world
        self.terminal_states = [self.state_to_index(goal_location) for goal_location in goal_locations]
        self.all_actions = [ACTION_UP, ACTION_DOWN, ACTION_LEFT, ACTION_RIGHT]

    def state_to_index(self, state):
        return state.x + state.y * self.world.width

    def index_to_state(self, state_index):
        return Coordinate(state_index % self.world.width, int(state_index / self.world.width))

    def get_random_initial_state_index(self):
        while True:
            random_coordinate = Coordinate(random.randrange(world.width), random.randrange(world.height))

            if world.get(random_coordinate) == EMPTY_TILE:
                return self.state_to_index(random_coordinate)

    def possible_actions(self, state_index):
        state = self.index_to_state(state_index)
        possible_actions = list()

        for (action_index, action) in enumerate(self.all_actions):
            if self.world.is_inside(state + action):
                if self.world.get(state + action) == EMPTY_TILE or self.world.get(state + action) == GOAL_TILE:
                    possible_actions.append(action_index)

        return possible_actions

    def perform_action(self, state_index, action_index):
        state = self.index_to_state(state_index)
        target_state = state + self.all_actions[action_index]

        return self.state_to_index(target_state)

    def reward(self, state_index, action_index):
        target_state = self.perform_action(state_index, action_index)

        if target_state in self.terminal_states:
            return 1000

        return -1

    def is_terminal_state(self, state_index):
        return state_index in self.terminal_states


def color_mapping(state):
    if state == WALL_TILE:
        return 0, 0, 0

    if state == GOAL_TILE:
        return 255, 0, 0

    if state == EMPTY_TILE:
        return 255, 255, 255

    return 0, 255, 0


if __name__ == "__main__":
    world = GridWorld(20, 20)
    world.fill(EMPTY_TILE)
    world.set(Coordinate(10, 1), GOAL_TILE)
    world.set(Coordinate(19, 10), GOAL_TILE)
    world.set(Coordinate(5, 5), GOAL_TILE)

    qLearning = QLearningGridWorld(world, [Coordinate(10, 1), Coordinate(19, 10), Coordinate(5, 5)])
    optimal_policies = list()
    images = list()
    optimal_policies.append(qLearning.get_optimal_policy())

    for policy_progress in range(75):
        print(policy_progress)
        qLearning.perform_q_value_iteration(40, print_debug=False)
        optimal_policies.append(qLearning.get_optimal_policy())

    for optimal_policy in optimal_policies:
        image = draw_world(world, color_mapping)
        action_representation = [GridWorldDrawer.ARROW_UP, GridWorldDrawer.ARROW_DOWN, GridWorldDrawer.ARROW_LEFT, GridWorldDrawer.ARROW_RIGHT]

        for x in range(world.width):
            for y in range(world.height):
                best_action = optimal_policy[qLearning.state_to_index(Coordinate(x, y))]
                draw_shape(image, world, x, y, action_representation[int(best_action)], (255, 0, 255))

        images.append(image)

    images[0].save("result.gif", save_all=True, append_images=images[1:], loop=0, optimize=False, quality=100, duration=200)

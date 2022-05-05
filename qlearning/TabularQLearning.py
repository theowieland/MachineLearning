from abc import abstractmethod

import numpy as np


class TabularQLearning:

    def __init__(self, num_states, num_actions, discount_factor, learning_rate):
        self.num_states = num_states
        self.num_actions = num_actions
        self.discount_factor = discount_factor
        self.learning_rate = learning_rate

        self.q_values = np.zeros(shape=(self.num_states, self.num_actions))
        self.epsilon = .1  # used for the epsilon greedy policy

    @abstractmethod
    def state_to_index(self, state):
        pass

    @abstractmethod
    def index_to_state(self, state_index):
        pass

    @abstractmethod
    def get_random_initial_state_index(self):
        pass

    @abstractmethod
    def possible_actions(self, state_index):
        pass

    @abstractmethod
    def perform_action(self, state_index, action_index):
        pass

    @abstractmethod
    def reward(self, state_index, action_index):
        pass

    @abstractmethod
    def is_terminal_state(self, state_index):
        pass

    def q_value(self, state_index, action_index):
        return self.q_values[state_index][action_index]

    def optimal_action(self, state_index):
        return np.argmax(self.q_values[state_index])

    def epsilon_greedy_policy(self, state_index, possible_actions):
        if np.random.uniform(0, 1) < self.epsilon:
            # select random action
            return possible_actions[np.random.randint(0, len(possible_actions))]

        # return the currently best action that is possible
        best_action_index = possible_actions[0]  # assumes that at least one action is always possible

        for possible_action in possible_actions:
            if self.q_values[state_index][possible_action] > self.q_values[state_index][best_action_index]:
                best_action_index = possible_action

        return best_action_index

    def perform_q_value_iteration(self, num_iterations=1000, print_debug=True):
        debug_steps = num_iterations / 25

        for iteration in range(num_iterations):
            if (iteration % debug_steps) == 0 and print_debug:
                print("q_value_iteration progress: %d%%" % ((iteration / num_iterations) * 100))

            current_state = self.get_random_initial_state_index()
            self.epsilon = max(.1, 1 - (iteration / num_iterations))

            while not self.is_terminal_state(current_state):
                possible_actions = self.possible_actions(current_state)
                selected_action = self.epsilon_greedy_policy(current_state, possible_actions)

                target_state = self.perform_action(current_state, selected_action)

                delta = self.reward(current_state, selected_action) - self.q_value(current_state, selected_action)
                if not self.is_terminal_state(target_state):
                    delta += self.discount_factor * self.q_value(target_state, self.optimal_action(target_state))

                self.q_values[current_state][selected_action] += self.learning_rate * delta
                current_state = target_state

    def get_optimal_policy(self):
        optimal_policy = np.zeros(self.num_states)

        for state_index in range(self.num_states):
            optimal_policy[state_index] = np.argmax(self.q_values[state_index])

        return optimal_policy






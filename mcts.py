import math
import copy
import numpy as np
from utils import ternary_to_decimal, select_action, select_random_action


def uct(reward_i: int, n_i: int, n_total: int, c: float):
    return reward_i / n_i + c * math.sqrt(math.log(n_total) / n_i)


class MCTS(object):

    def __init__(self, env, player: int, rollout_factor: int, uct_c_constant: float):
        self.env = env
        self.rollout_factor = rollout_factor # rollouts =  rollout_factor x number_sub_nodes
        self.uct_c_constant = uct_c_constant
        self.player = player # -1 for naught, +1 for cross


    def get_next_action(self):
        self._reset_vars()
        for _ in range(self.rollout_factor * len(self.sub_states)):
            sub_state = self.selection()
            reward = self.rollout()
            self.update_statistics(sub_state, reward)
        return self._most_visited_state()


    def selection(self):
        if len(self.unvisited_sub_states) > 0:
            sub_state = self.unvisited_sub_states.pop()
        else:
            sub_state = self._best_uct()
        return sub_state


    def rollout(
            self,
            action_type_naught='random',
            action_type_cross='random',
            Q_naught=None,
            Q_cross=None,
            epsilon=None,
            verbosity=False
):

        # save current state
        env_attribute_state = copy.deepcopy(self.env.__dict__)

        # rollout
        done = False
        while not done:

            # state
            s_hash = self.env.getHash()
            s = ternary_to_decimal(s_hash)

            # select action
            available_actions = self.env.getEmptySpaces()
            available_actions_int_lst = [self.env.int_from_action(i) for i in available_actions]
            if self.env.curTurn == -1:
                a = select_action(action_type_naught, Q_naught, s, available_actions_int_lst, epsilon)
            elif self.env.curTurn == 1:
                a = select_action(action_type_cross, Q_cross, s, available_actions_int_lst, epsilon)

            # make action
            observation, reward, done, info = self.env.step(self.env.action_from_int(a), verbosity)

        # return state before roll out
        self.env.__dict__ = env_attribute_state

        return reward

    def update_statistics(self, state, reward):
        # score update
        if self.score_dct.get(state) is None:
            self.score_dct[state] = {-1: 0, 0: 0, 1: 0}
            self.score_dct[state][reward] += 1
        else:
            self.score_dct[state][reward] += 1

        # uct update
        reward_i = self.score_dct[state][self.player] - self.score_dct[state][-self.player]
        n_i = sum(self.score_dct[state].values())
        n_total = sum([sum([quantity for quantity in wins_stat.values()]) for wins_stat in self.score_dct.values()])
        c = self.uct_c_constant
        self.uct_dct[state] = uct(reward_i, n_i, n_total, c)

    def _most_visited_state(self):
        most_visited = max([sum([quantity for quantity in wins_stat.values()]) for wins_stat in self.score_dct.values()])
        for state in self.score_dct.keys():
            if sum(self.score_dct[state].values()) == most_visited:
                return state

    def _reset_vars(self):
        # обнуление переменных перед новым действием
        self.sub_states = [self.env.int_from_action(i) for i in self.env.getEmptySpaces()]
        self.unvisited_sub_states = [self.env.int_from_action(i) for i in self.env.getEmptySpaces()]
        self.score_dct = {} # {state_1{-1: n, 0: m, 1: k}, state_2: {аналогично} ... state_n:{аналогично}}
        self.uct_dct = {} # {state_1: uct_1, state_2: uct_2, ... state_n: uct_n}

    def _best_uct(self):
        max_uct = max(self.uct_dct.values())
        for state in self.uct_dct.keys():
            if self.uct_dct[state] == max_uct:
                return state





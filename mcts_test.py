import math

import pytest

from env import TicTacToe
from mcts import MCTS

def uct(reward_i: int, n_i: int, n_total: int, c: float):
    return reward_i / n_i + c * math.sqrt(math.log(n_total) / n_i)

# env
N_ROWS = 3
N_COLS = 3
N_WINS = 3
env = TicTacToe(n_rows=N_ROWS, n_cols=N_COLS, n_win=N_WINS)

# msts
PLAYER = 1
ROLLOUT_FACTOR = 2
UCT_C_CONSTANT = 0.3
mcts_strategy = MCTS(env=env, player=PLAYER, rollout_factor=ROLLOUT_FACTOR, uct_c_constant=UCT_C_CONSTANT)


def test_init():
    assert mcts_strategy.player == PLAYER
    assert mcts_strategy.rollout_factor == ROLLOUT_FACTOR
    assert mcts_strategy.uct_c_constant == UCT_C_CONSTANT

def test_reset_vars():
    mcts_strategy._reset_vars()
    assert mcts_strategy.sub_states == [0, 1, 2, 3, 4, 5, 6, 7, 8]
    assert mcts_strategy.unvisited_sub_states == [0, 1, 2, 3, 4, 5, 6, 7, 8]
    assert mcts_strategy.score_dct == {}
    assert mcts_strategy.uct_dct == {}

def test_selection_if_unvisited():
    sub_state = mcts_strategy.selection()
    assert sub_state == 8
    assert mcts_strategy.sub_states == [0, 1, 2, 3, 4, 5, 6, 7, 8]
    assert mcts_strategy.unvisited_sub_states == [0, 1, 2, 3, 4, 5, 6, 7]

    # еще 8 вершин у
    for i in range(8):
        sub_state = mcts_strategy.selection()
    assert sub_state == 0
    assert mcts_strategy.sub_states == [0, 1, 2, 3, 4, 5, 6, 7, 8]
    assert mcts_strategy.unvisited_sub_states == []

def test_update_statistics_id_does_not_exist_in_dict():
    mcts_strategy.update_statistics(0, -1)
    assert mcts_strategy.score_dct == {0: {-1: 1, 0: 0, 1: 0}}
    assert mcts_strategy.uct_dct == {0: -1.0}

    mcts_strategy.update_statistics(1, -1)
    mcts_strategy.update_statistics(2, -1)

    mcts_strategy.update_statistics(3, 0)
    mcts_strategy.update_statistics(4, 0)
    mcts_strategy.update_statistics(5, 0)

    mcts_strategy.update_statistics(6, 1)
    mcts_strategy.update_statistics(7, 1)
    mcts_strategy.update_statistics(8, 1)

    # вершина 0
    assert mcts_strategy.uct_dct[0] == uct(reward_i=-1, n_i=1, n_total=1, c=UCT_C_CONSTANT)
    # вершина 1
    assert mcts_strategy.uct_dct[8] == uct(reward_i=1, n_i=1, n_total=9, c=UCT_C_CONSTANT)

def test_selection_by_uct():
    sub_state = mcts_strategy.selection()
    assert sub_state == 8

def update_statistics_id_exists_in_dict():
    mcts_strategy.update_statistics(8, -1)
    assert mcts_strategy.score_dct[8] == {-1: 1, 0: 0, 1: 1}
    assert mcts_strategy.uct_dct[8] == uct(reward_i=0, n_i=2, n_total=10, c=UCT_C_CONSTANT)



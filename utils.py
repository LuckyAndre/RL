import numpy as np
import matplotlib.pyplot as plt
import tqdm


def decimal_to_ternary(n: int):
    if n == 0:
        return '0'
    nums = []
    while n:
        n, r = divmod(n, 3)
        nums.append(str(r))
    return ''.join(reversed(nums))


def ternary_to_decimal(n: str):
    n = int(n)
    if n == 0:
        return 0
    num = 0
    i = 0 # разряд
    while n:
        n, r = divmod(n, 10)
        num += r * pow(3, i)
        i += 1
    return num


# def get_random_Q(Q_space_size):
#     """
#     Создание случайной функции Q
#     """
#     #Q = np.random.random(size=Q_space_size)
#     Q = np.random.default_rng().random(Q_space_size, dtype='float32')
#     return Q


# def compute_policy_by_Q(Q):
#     """
#     Вычисление стратегии: для каждого состояния выбираем действие с максимальным значением Q-функции
#     """
#     return np.argmax(Q, axis=(len(Q.shape) - 1))


def select_random_action(available_actions_int_lst: list):
    """
    Случайный выбор действия из доступных
    """
    if len(available_actions_int_lst) > 0:
        rand_value = np.random.randint(len(available_actions_int_lst))
        return available_actions_int_lst[rand_value]
    else:
        return None


def select_greedy_action(Q: np.array, state: int, available_actions_int_lst: list):
    """
    Жадный выбор действия из доступных
    """
    try:
        available_Q_values = np.array([Q[state][action] for action in available_actions_int_lst])
        best_action = np.argmax(available_Q_values)
        return available_actions_int_lst[best_action]
    except:
        # возможно попадание в состояние, для которого еще нет статистики - тогда действуем случайно
        return select_random_action(available_actions_int_lst)


def select_soft_action(Q: np.array, state: int, available_actions_int_lst: list, epsilon: float):
    """
    Мягкий выбор действия из доступных
    """
    a = select_greedy_action(
        Q,
        state,
        available_actions_int_lst
    ) if np.random.rand() > epsilon else select_random_action(available_actions_int_lst)
    return a


def select_action(action_type: str, Q: np.array, s: int, available_actions_int_lst: list, epsilon: float, mcts=None, env=None):
    """
    Выбор типа действия
    action type: "random" -> случайное, "greedy" -> жадное, "exploration" -> с исследованием, "mcts" -> monte carlo tree search
    """
    if action_type == 'random':
        return select_random_action(available_actions_int_lst)
    elif action_type == 'greedy':
        return select_greedy_action(Q, s, available_actions_int_lst)
    elif action_type == 'exploration':
        return select_soft_action(Q, s, available_actions_int_lst, epsilon)
    elif action_type == 'mcts':
        mcts.env = env
        return mcts.get_next_action()
    else:
        raise ValueError(f'{action_type} is not valid value. Action type could only be "random", "greedy", "exploration" or "mcts".')


def run_episode_Q_learning(
        action_type_naught: str, action_type_cross: str,
        env, Q_naught: dict, Q_cross: dict, n_action: int,
        learner: int,
        alpha=0.05, epsilon=0.0, gamma=0.9,
        verbosity=False
):

    """
    Переоценка функции Q на основе одного эпизода.
    action_type_naught, action_type_cross: "random", "greedy" or "exploration"
    n_action: число действий (для задания функции Q)
    learner: 1 если обучаем крестиков, -1 если обучаем ноликов
    """
    done = False
    env.reset()

    # state
    s_hash = env.getHash()
    s = ternary_to_decimal(s_hash)

    # action
    available_actions = env.getEmptySpaces()
    available_actions_int_lst = [env.int_from_action(i) for i in available_actions]

    # first action makes cross
    if Q_cross is not None:
        if Q_cross.get(s) is None: # haven't visited yet?
            Q_cross[s] = np.random.default_rng().random(n_action, dtype='float32')
    a = select_action(action_type_cross, Q_cross, s, available_actions_int_lst, epsilon)

    # if learner is naught we should get his first state and action
    if learner == -1:
        # cross make step
        observation, reward, done, info = env.step(env.action_from_int(a), verbosity)

        # state
        s_hash = observation[0]
        s = ternary_to_decimal(s_hash)

        # action
        available_actions = env.getEmptySpaces()
        available_actions_int_lst = [env.int_from_action(i) for i in available_actions]

        if Q_naught is not None:
            if Q_naught.get(s) is None:  # haven't visited yet?
                Q_naught[s] = np.random.default_rng().random(n_action, dtype='float32')
        a = select_action(action_type_naught, Q_naught, s, available_actions_int_lst, epsilon)

    while not done:
        # make step. Return: (hash, empty, cur_turn), reward, gameFinished?, {}
        observation, reward, done, info = env.step(env.action_from_int(a), verbosity)

        # OPPONENT TURN
        # state
        s_opponent_hash = observation[0]
        s_opponent = ternary_to_decimal(s_opponent_hash)

        # action
        available_actions = observation[1]
        available_actions_int_lst = [env.int_from_action(i) for i in available_actions]
        if not done:

            if env.curTurn == -1:
                if Q_naught is not None:
                    if Q_naught.get(s_opponent) is None:  # haven't visited yet?
                        Q_naught[s_opponent] = np.random.default_rng().random(n_action, dtype='float32')
                a_opponent = select_action(action_type_naught, Q_naught, s_opponent, available_actions_int_lst, epsilon)

            elif env.curTurn == 1:
                if Q_cross is not None:
                    if Q_cross.get(s_opponent) is None:  # haven't visited yet?
                        Q_cross[s_opponent] = np.random.default_rng().random(n_action, dtype='float32')
                a_opponent = select_action(action_type_cross, Q_cross, s_opponent, available_actions_int_lst, epsilon)

            observation, reward, done, info = env.step(env.action_from_int(a_opponent), verbosity)

        else:
            continue

        # LEARNER TURN
        # next state
        s_hash = observation[0]
        s_prime = ternary_to_decimal(s_hash)

        # next action
        available_actions = observation[1]
        available_actions_int_lst = [env.int_from_action(i) for i in available_actions]

        if env.curTurn == -1:
            if Q_naught is not None:
                if Q_naught.get(s_prime) is None:  # еще не посещали?
                    Q_naught[s_prime] = np.random.default_rng().random(n_action, dtype='float32')
            a_prime = select_action(action_type_naught, Q_naught, s_prime, available_actions_int_lst, epsilon)
            Q_naught[s][a] = Q_naught[s][a] + alpha * (-reward + gamma * np.max(Q_naught[s_prime]) - Q_naught[s][a])

        elif env.curTurn == 1:
            if Q_cross is not None:
                if Q_cross.get(s_prime) is None:  # еще не посещали?
                    Q_cross[s_prime] = np.random.default_rng().random(n_action, dtype='float32')
            a_prime = select_action(action_type_cross, Q_cross, s_prime, available_actions_int_lst, epsilon)
            Q_cross[s][a] = Q_cross[s][a] + alpha * (reward + gamma * np.max(Q_cross[s_prime]) - Q_cross[s][a])

        s, a = s_prime, a_prime

    return Q_naught, Q_cross, reward


def test_game(
        action_type_naught, action_type_cross,
        env, Q_naught, Q_cross,
        epsilon,
        verbosity=False,
        mcts=None,
):
    done = False
    env.reset()

    while not done:

        # state
        s_hash = env.getHash()
        s = ternary_to_decimal(s_hash)

        # select action
        available_actions = env.getEmptySpaces()
        available_actions_int_lst = [env.int_from_action(i) for i in available_actions]
        if env.curTurn == -1:
            a = select_action(action_type_naught, Q_naught, s, available_actions_int_lst, epsilon, mcts, env)
        elif env.curTurn == 1:
            a = select_action(action_type_cross, Q_cross, s, available_actions_int_lst, epsilon, mcts, env)

        # make action
        observation, reward, done, info = env.step(env.action_from_int(a), verbosity)

    env.close()
    return reward


def measure_metrics(
        n_episodes,
        action_type_naught, action_type_cross,
        env, Q_naught, Q_cross,
        epsilon,
        verbosity=False,
        mcts=None,
):
    metrics = {-1: 0, 0: 0, 1: 0}
    for n in range(n_episodes):
        winner = test_game(
            action_type_naught, action_type_cross,
            env, Q_naught, Q_cross,
            epsilon,
            verbosity=False,
            mcts=mcts
        )
        metrics[winner] += 1

    if verbosity:
        print(
            f"naught wins = {round(metrics[-1] / sum(metrics.values()), 2)}",
            f"\ncross wins = {round(metrics[1] / sum(metrics.values()), 2)}",
            f"\ndraws = {round(metrics[0] / sum(metrics.values()), 2)}\n")

    return metrics


def plot_metric(metrics_lst, print_step):
    # figure
    plt.figure(figsize=(20, 5))
    ax = plt.axes()  # Add an axes to the current figure and make it the current axes

    # data
    naught = [element[-1] / sum(element.values()) for element in metrics_lst]
    cross = [element[1] / sum(element.values()) for element in metrics_lst]
    draws = [element[0] / sum(element.values()) for element in metrics_lst]

    # plot
    x_range = np.array(range(len(metrics_lst))) * print_step
    ax.plot(x_range, naught, label='naught')
    ax.plot(x_range, cross, label='cross')
    ax.plot(x_range, draws, label='draws')

    # annotation
    ax.set_title('Мониторинг обучения', fontsize=14, )
    ax.set_xlabel('эпизод')
    ax.set_xticks(x_range)
    _ = ax.legend()
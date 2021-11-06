from itertools import product

import numpy as np
import matplotlib.pyplot as plt


def observation_int(observation):
    """
    перевод ответа среды в int
    """
    return tuple([int(i) for i in observation])


def run_episode(env, pi):
    """
    прогон одного эпизода в среде env по стратегии pi с сохранением пройденных состояний и наград
    """
    # начальное состояние
    observation = env.reset()
    states, rewards = [observation_int(observation)], [0.0]  # S0, A0, R1, S1, A1... (R0 считаем равным 0)

    # шаги по стратегии
    for _ in range(1000):
        observation, reward, done, info = env.step(pi[states[-1]])
        states.append(observation_int(observation))
        rewards.append(reward)

        if done:
            break

    return states, rewards


def update_returns(R_sum, R_count, states, rewards, gamma):
    """
    Для одного эпизода (серия состояний states) расчитываем награду с учетом дисконтирования
    """
    # шагаем в обратном порядке, пропустив терминальную вершину
    g = 0
    for t in range(len(states) - 2, -1, -1):
        g = g * gamma + rewards[t + 1]  # Gt = R(t+1) + γ* R(t+2) + γ^2 * R(t+3)+… далее вычислим vπ(s)=𝔼π[Gt∣St=s]
        R_sum[states[t]] += g
        R_count[states[t]] += 1

    return R_sum, R_count


def score_pi(env, pi, episodes=100_000):
    """
    Средняя оценка стратегии pi в среде env
    """
    total_score = 0
    for _ in range(episodes):
        states, rewards = run_episode(env, pi)
        total_score += sum(rewards)
    return total_score / episodes


def get_random_Q(Q_space_size):
    """
    Создание случайной функции Q
    """
    Q = np.random.random(size=(Q_space_size))

    # player not usable states
    Q[0:5] = 0.0

    # player terminal states
    Q[22:] = 0.0

    # dealer not usable states
    Q[:, 0] = 0.0
    return Q


def compute_policy_by_Q(Q):
    """
    Вычисление стратегии: для каждого состояния выбираем действие с максимальным значением Q-функции
    """
    return np.argmax(Q, axis=(len(Q.shape) - 1))


def run_episode_Q_learning(env, pi, Q, alpha=0.05, epsilon=0.0, gamma=0.9):
    """
    Переоценка функции Q на основе одного эпизода
    """
    nA = Q.shape[-1]

    observation = env.reset()
    s = observation_int(observation)  # S0
    a = pi[s] if np.random.rand() > epsilon else np.random.randint(nA)  # A0

    for _ in range(1000):
        observation, reward, done, info = env.step(a)  # шаг
        s_prime = observation_int(observation)  # следующее состояние
        if observation[3] < 0:
            print('less 0', observation, s_prime)
        a_prime = pi[s_prime] if np.random.rand() > epsilon else np.random.randint(nA)  # следующее действие
        Q[s][a] = Q[s][a] + alpha * (reward + gamma * np.max(Q[s_prime]) - Q[s][a])  # оценка Q с учетом след. состояния
        s, a = s_prime, a_prime

        if done:
            break

    return Q


def params_grid(gam_lst=[1], alp_lst=[0.5, 0.1, 0.15], eps_lst=[0.5, 0.1, 0.15]):
    """
    Сетка подбора гиперпараметров
    """
    for combination in product(gam_lst, alp_lst, eps_lst):
        param = {
            'gamma': combination[0],
            'alpha': combination[1],
            'epsilon': combination[2],
        }
        yield param


def learning_loop(env, pi, Q, alpha, epsilon, gamma, total_episodes, pi_score_step):
    """
    Итератирвно улучшаем стратегию pi повторив total_episodes шагов Q-learning с параметрами alpha, epsilon, gamma
    """
    # переменные для построения графиков
    score_results = []
    score_step = []

    # цикл обучения и оценки
    for n in range(total_episodes):

        # оценка стратегии
        if (n % pi_score_step) == 0:
            score_results.append(score_pi(env, pi, episodes=1_000))
            score_step.append(n)

        # Q-learning step
        Q = run_episode_Q_learning(env=env, pi=pi, Q=Q, alpha=alpha, epsilon=epsilon, gamma=gamma)
        pi = compute_policy_by_Q(Q)

    return {
        'pi': pi,
        'Q': Q,
        'score_step': score_step,
        'score_results': score_results,
        'alpha': alpha,
        'epsilon': epsilon,
        'gamma': gamma
    }


def plot_learning_curve(learning_results_lst):
    """
    Отрисовка результатов обучения, полученный с помощью функции learning_loop
    """
    # figure
    fig = plt.figure(figsize=(20, 8))
    ax = plt.axes() # Add an axes to the current figure and make it the current axes

    # data
    for l_result in learning_results_lst:
        ax.plot(
            l_result['score_step'],
            l_result['score_results'],
            label=f"alp={l_result['alpha']} eps={l_result['epsilon']} gam={l_result['gamma']}"
        )

    # annotation
    ax.set_title('Learning speed', fontsize=14, fontweight='bold')
    ax.set_xlabel('episodes')
    ax.set_ylabel('mean reward')
    ax.legend()
    plt.show(fig)


def run_episode_actions(env, pi, nA, epsilon):
    """
    прогон одного эпизода в среде env по стратегии pi с сохранением пройденных состояний, действий и наград
    """

    # начальное состояние
    observation = env.reset()
    s = observation_int(observation)  # S0
    a = pi[s] if np.random.rand() > epsilon else np.random.randint(nA)  # A0

    # списки с состояниями, действиями, наградами
    states, actions, rewards = [s], [a], [0]

    # шаги по стратегии
    for _ in range(1000):
        observation, reward, done, info = env.step(a)
        s = observation_int(observation)
        states.append(s)
        a = pi[s] if np.random.rand() > epsilon else np.random.randint(nA)
        actions.append(a)
        rewards.append(reward)
        if done:
            break

    return states, actions, rewards


def update_returns_actions(R_sum, R_count, states, actions, rewards, gamma):
    """
    Для одного эпизода (серия состояний states) расчитываем награду с учетом дисконтирования
    """
    # шагаем в обратном порядке, пропустив терминальную вершину
    g = 0
    for t in range(len(states) - 2, -1, -1):
        g = g * gamma + rewards[t + 1]  # Gt = R(t+1) + γ* R(t+2) + γ^2 * R(t+3)+… далее вычислим vπ(s)=𝔼π[Gt∣St=s]
        R_sum[states[t]][actions[t]] += g
        R_count[states[t]][actions[t]] += 1

    return R_sum, R_count


def update_returns_actions_offpolicy_MC(Q, C, pi, states, actions, rewards, epsilon, gamma):
    """
    states, actions, rewards сгененрированы по мягкой стратегии pi'
    """

    # инициализирование
    nA = Q.shape[-1]
    g = 0
    w = 1
    prob_best_action = 1 - (nA - 1) * epsilon / nA  # ???

    # обновление Q
    s = states[-1]
    a = actions[-1]
    C[s][a] = C[s][a] + w
    Q[s][a] = Q[s][a] + w / C[s][a] * (g - Q[s][a]) # формула итеративного обновления с делением на сумму весов

    for t in range(len(states) - 2, -1, -1):

        if actions[t + 1] != pi[states[t + 1]]:  # если действие pi' расходится со стратегией pi
            break

        g = g * gamma + rewards[t + 1]
        w = w * 1 / (prob_best_action)

        # обновление Q
        s, a = states[t], actions[t]
        C[s][a] = C[s][a] + w
        Q[s][a] = Q[s][a] + w / C[s][a] * (g - Q[s][a])

    return Q, C


def params_grid_MC(gam_lst=[1], eps_lst=[0.05, 0.1, 0.3, 0.5, 0.7]):
    """
    Сетка подбора гиперпараметров
    """
    for combination in product(gam_lst, eps_lst):
        param = {
            'gamma': combination[0],
            'epsilon': combination[1],
        }
        yield param


def learning_loop_MC(env, pi, Q, C, epsilon, gamma, total_episodes, pi_score_step):
    """
    Итератирвно улучшаем стратегию pi повторив total_episodes шагов offpolicy_MC epsilon, gamma
    """
    # переменные для построения графиков
    score_results = []
    score_step = []
    # цикл обучения и оценки
    for n in range(total_episodes):

        # оценка стратегии
        if (n % pi_score_step) == 0:
            score_results.append(score_pi(env, pi, episodes=1_000))
            score_step.append(n)

        # offpolicy_MC step
        nA = Q.shape[-1]
        states, actions, rewards = run_episode_actions(env, pi, nA, epsilon) # гененрируем случайный эпизод
        Q, C = update_returns_actions_offpolicy_MC(Q, C, pi, states, actions, rewards, epsilon, gamma)
        pi = compute_policy_by_Q(Q)

    return {
        'pi': pi,
        'Q': Q,
        'score_step': score_step,
        'score_results': score_results,
        'epsilon': epsilon,
        'gamma': gamma
    }


def plot_learning_curve_MC(learning_results_lst):
    """
    Отрисовка результатов обучения, полученный с помощью функции learning_loop_MC
    """
    # figure
    fig = plt.figure(figsize=(20, 8))
    ax = plt.axes() # Add an axes to the current figure and make it the current axes

    # data
    for l_result in learning_results_lst:
        ax.plot(
            l_result['score_step'],
            l_result['score_results'],
            label=f"eps={l_result['epsilon']} gam={l_result['gamma']}"
        )

    # annotation
    ax.set_title('Learning speed', fontsize=14, fontweight='bold')
    ax.set_xlabel('episodes')
    ax.set_ylabel('mean reward')
    ax.legend()
    plt.show(fig)
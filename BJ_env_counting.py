import gym
from gym import spaces
from gym.utils import seeding


CARD_SCORE = {
    1: -2,
    2: 1,
    3: 2,
    4: 2,
    5: 3,
    6: 2,
    7: 1,
    8: 0,
    9: -1,
    10: -2
}


def cmp(a, b):
    return float(a > b) - float(a < b)


# карта
def draw_card(np_random, deck):
    return deck.pop(np_random.choice(range(len(deck)))) # карта извлекается из колоды


# Туз == 11?
def usable_ace(hand):  # Does this hand have a usable ace?
    return 1 in hand and sum(hand) + 10 <= 21


def sum_hand(hand):  # Return current hand total
    """
    В текущей реализации если туз можно засчитать за 11 - он будет засчитан за 11!
    Пример эпизода, в ходе раздачи было получено 4 туза и они считались по разному в зависимости от общей суммы карт:
    [2 + 3,      1,          1,          1,          8,          1,          10]
    [(5, 9, 0), (16, 9, 1), (17, 9, 1), (18, 9, 1), (16, 9, 0), (17, 9, 0), (27, 9, 0)]
    """
    if usable_ace(hand):
        return sum(hand) + 10
    return sum(hand)


def is_bust(hand):  # Is this hand a bust?
    return sum_hand(hand) > 21


def score(hand):  # What is the score of this hand (0 if bust)
    return 0 if is_bust(hand) else sum_hand(hand)


# получил туза + 10 = 21 с раздачи
def is_natural(hand):  # Is this hand a natural blackjack?
    return sorted(hand) == [1, 10]


class BlackjackCountingEnv(gym.Env):

    def __init__(self, natural=False, sab=False):
        self.deck = []
        self.reset_deck()
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Tuple((
            spaces.Discrete(32), # player sum
            spaces.Discrete(11), # dealer first card
            spaces.Discrete(2), # usable ase = True or False
            spaces.Discrete(89) # sum of deck_out cards: could change from -44 to + 44
        ))
        self.seed()
        self.natural = natural # Flag to payout 1.5 on a "natural" blackjack win, like casino rules
        self.sab = sab # Flag for full agreement with the (Sutton and Barto, 2018) definition. Overrides self.natural

    def reset_deck(self, start=False):
        if len(self.deck) < 15:
            # 1 = Ace, 2-10 = Number cards, Jack/Queen/King = 10
            self.deck = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 10, 10] * 4
            self.deck_out = []

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step_hit(self):
        card = draw_card(self.np_random, self.deck)
        self.deck_out.append(card)
        self.player.append(card)
        if is_bust(self.player):
            done = True
            reward = -1.0
        else:
            done = False
            reward = 0.0
        return done, reward

    def step_stick(self):
        done = True

        # диллер открывает свою скрытую карту
        self.deck_out.append(self.card_d_2)

        # добирает карты из колоды
        while sum_hand(self.dealer) < 17:
            card = draw_card(self.np_random, self.deck)
            self.deck_out.append(card)
            self.dealer.append(card)

        reward = cmp(score(self.player), score(self.dealer))

        # если (sab) и (21 у игрока с раздачи) и (не 21 у дилера с раздачи), то:
        if self.sab and is_natural(self.player) and not is_natural(self.dealer):
            # Player automatically wins. Rules consistent with S&B
            reward = 1.0

        # если (не sab) и (натуральный) и (21 у игрока с раздачи) и (у дилера любым способом меньше), то:
        elif (
                not self.sab
                and self.natural
                and is_natural(self.player)
                and reward == 1.0
        ):
            # Natural gives extra points, but doesn't autowin. Legacy implementation
            reward = 1.5

        return done, reward

    def step_double(self):
        card = draw_card(self.np_random, self.deck)
        self.deck_out.append(card)
        self.player.append(card)
        done = True
        if is_bust(self.player):
            reward = -1.0
        else:
            _, reward = self.step_stick()
        return done, reward * 2

    def step(self, action):

        assert self.action_space.contains(action)

        if action == 1:  # hit
            done, reward = self.step_hit()

        elif action == 2:  # double
            done, reward = self.step_double()

        elif action == 0:  # stick
            done, reward = self.step_stick()

        return self._get_obs(), reward, done, {}

    def _get_obs(self):
        return (
            sum_hand(self.player),
            self.dealer[0],
            usable_ace(self.player),
            sum([CARD_SCORE[card] for card in self.deck_out]) + 44 # положительные значения для корректной индексации
        )

    def reset(self):

        # проверяем колоду
        self.reset_deck()

        # раздача стартовых карт
        card_d_1 = draw_card(self.np_random, self.deck)
        self.card_d_2 = draw_card(self.np_random, self.deck) # эту карту узнаю чуть позже, когда диллер откроется!
        card_p_1 = draw_card(self.np_random, self.deck)
        card_p_2 = draw_card(self.np_random, self.deck)
        self.dealer = [card_d_1, self.card_d_2]
        self.player = [card_p_1, card_p_2]

        # в подсчете "теплоты" колоды учитываем только первую карту дилера (вторая пока перевернута)
        self.deck_out = self.deck_out + [card_d_1, card_p_1, card_p_2]

        return self._get_obs()
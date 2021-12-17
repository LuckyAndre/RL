import gym
import numpy as np
import matplotlib.pyplot as plt


N_ROWS, N_COLS, N_WIN = 3, 3, 3


class TicTacToe(gym.Env):
    def __init__(self, n_rows=N_ROWS, n_cols=N_COLS, n_win=N_WIN, clone=None):
        # n_wins - сколько подряд идущих значений должно быть для победы
        if clone is not None: # не разбирал (вроде, он не запускается)
            self.n_rows, self.n_cols, self.n_win = clone.n_rows, clone.n_cols, clone.n_win
            self.board = copy.deepcopy(clone.board)
            self.curTurn = clone.curTurn
            self.emptySpaces = None
            self.boardHash = None
        else:
            self.n_rows = n_rows
            self.n_cols = n_cols
            self.n_win = n_win

            self.reset()

    def getEmptySpaces(self):
        if self.emptySpaces is None:
            res = np.where(self.board == 0) # (array(x1, x2,.. yn), array(y1, y2,.. yn)) координаты, удовл. условию
            self.emptySpaces = np.array([(i, j) for i, j in zip(res[0], res[1])])
        return self.emptySpaces # array([x1, y1], [x2, y2],.. [xn, yn])

    def makeMove(self, player, i, j): # отмечаю ход на поле
        self.board[i, j] = player
        self.emptySpaces = None
        self.boardHash = None

    def getHash(self):
        if self.boardHash is None:
            self.boardHash = ''.join(['%s' % (x + 1) for x in self.board.reshape(self.n_rows * self.n_cols)])
        return self.boardHash # строка вида (для 3х3) 211101111, где 2 - это крестик, 0 - это нолик, 1 = пусто

    def isTerminal(self):
        """
        Возвращается количество очков (нолики суммируются в минус, крестики - в плюс)
        -1 очко (победа "нолика")
        1 очко (победа "крестика")
        0 ничья
        None игра продолжается
        """

        cur_p = self.curTurn # чей ход
        cur_marks = np.where(self.board == self.curTurn) # какие у него позиции

        # проверим, не закончилась ли игра
        for i, j in zip(cur_marks[0], cur_marks[1]):
            win = False

            if i <= self.n_rows - self.n_win: # проверяем столбик
                if np.all(self.board[i:i + self.n_win, j] == cur_p):
                    win = True

            if not win: # проверяем строку
                if j <= self.n_cols - self.n_win:
                    if np.all(self.board[i, j:j + self.n_win] == cur_p):
                        win = True
            if not win:
                if i <= self.n_rows - self.n_win and j <= self.n_cols - self.n_win: # диагональ слева направо
                    if np.all(np.array([self.board[i + k, j + k] == cur_p for k in range(self.n_win)])):
                        win = True
            if not win:
                if i <= self.n_rows - self.n_win and j >= self.n_win - 1: # диагональ справа налево
                    if np.all(np.array([self.board[i + k, j - k] == cur_p for k in range(self.n_win)])):
                        win = True
            if win:
                self.gameOver = True
                return self.curTurn

        if len(self.getEmptySpaces()) == 0:
            self.gameOver = True
            return 0 # return 0 - это ничья, верно?

        self.gameOver = False
        return None

    def printBoard(self):
        print('╭', ('───┬' * self.n_cols)[:-1], '╮', sep='')
        for i in range(0, self.n_rows):
            if i != 0:
                print('├', ('───┼' * self.n_cols)[:-1], '┤', sep='')
            out = '│ '
            for j in range(0, self.n_cols):
                if self.board[i, j] == 1:
                    token = 'x'
                if self.board[i, j] == -1:
                    token = 'o'
                if self.board[i, j] == 0:
                    token = ' '
                out += token + ' │ '
            print(out, sep='')
        print('╰', ('───┴' * self.n_cols)[:-1], '╯', sep='')

    def getState(self):
        return (self.getHash(), self.getEmptySpaces(), self.curTurn)

    def action_from_int(self, action_int):
        # если последовательно пронумеровать поле, то в эту функцию можно передвать номер позиции
        # и он будет преобразовывать в tuple(x, y)
        # например для поля 3 х 3, 0 -> (0, 0), 3 -> (1, 0)
        return (int(action_int / self.n_cols), int(action_int % self.n_cols))

    def int_from_action(self, action):
        return action[0] * self.n_cols + action[1]

    def step(self, action, verbosity):
        """
        (hash, empty, cur_turn), reward, gameFinished?, {}
        """
        if self.board[action[0], action[1]] != 0:
            return self.getState(), -10, True, {} # - 10 очков штраф, если выбирается уже занятая позиция
        self.makeMove(self.curTurn, action[0], action[1])
        reward = self.isTerminal()
        self.curTurn = -self.curTurn
        if verbosity:
            self.printBoard()
        return self.getState(), 0 if reward is None else reward, reward is not None, {}

    def reset(self):
        self.board = np.zeros((self.n_rows, self.n_cols), dtype=int)
        self.boardHash = None
        self.gameOver = False
        self.emptySpaces = None
        self.curTurn = 1
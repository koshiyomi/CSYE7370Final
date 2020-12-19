import random

import gym
import numpy as np
import pandas as pd
from gym import spaces

from StockDataPreprocessor import StockDataPreprocessor

N_STOCK_INFO = 5
N_HISTORY = 100

ASSET = 1000000

BANK_INTEREST = 100
TRANSACTION_FEE = 10
DAILY_TRANSACTION_LIMIT = 5000
MAXIMUM_STEPS = 1000


class StockTrade(gym.Env):
    """
    Stock trade environment with continuous action space and modifiable stock quantity
    """

    def __init__(self, path='archive/Data/Stocks', stock_quantity=5, change_stocks=True):
        """
        initializer of StockTrade
        :param path: path where the stock data stored
        :param stock_quantity: 
        :param change_stocks:
        """

        # define action and observation space
        self.action_space = spaces.Box(low=-1, high=1, shape=(stock_quantity,))

        # observation space with current stock price, stock hold info, and stock price history
        observation_space_size = stock_quantity * N_STOCK_INFO + \
                                 stock_quantity + \
                                 N_HISTORY * stock_quantity * N_STOCK_INFO
        self.observation_space = spaces.Box(low=0, high=np.inf, shape=(observation_space_size,))
        self.reward_range = (-float('inf'), float('inf'))

        # generate stock data using
        self.sdp = StockDataPreprocessor(path, stock_quantity)

        # store dataframe for data visualization
        self.pd_data = self.sdp.get_data()

        # numpy for presenting states
        self.np_data = (self.pd_data.drop('Date', axis=1)).to_numpy()

        # environment variables
        self.stock_quantity = stock_quantity
        self.change_stocks = change_stocks

        # in game variables
        self.starting_day = random.randrange(len(self.pd_data) - MAXIMUM_STEPS)
        self.current_day = self.starting_day
        self.stock_history = np.zeros(N_HISTORY * stock_quantity * N_STOCK_INFO)

        # stock variables
        self.stock_hold = np.zeros(stock_quantity)
        self.current_asset = ASSET

        # game status variable
        self.done = False

    def step(self, action):
        if not self.done:
            current_price = self.np_data[self.current_day]
            current_history = self.stock_history
            self.stock_history = np.roll(self.stock_history, self.stock_quantity * N_STOCK_INFO)

            for i in range(len(current_price)):
                self.stock_history[i] = current_price[i]

            reward = 0

            # calculate reward of stock transaction
            for i in range(self.stock_quantity):
                stock_share_amount = action[i] * DAILY_TRANSACTION_LIMIT
                high_price = current_price[N_STOCK_INFO * i + 1]
                low_price = current_price[N_STOCK_INFO * i + 2]

                # buy stock
                if stock_share_amount > 0:
                    self.stock_hold[i] += stock_share_amount
                    stock_price = - low_price * stock_share_amount - TRANSACTION_FEE
                    reward += stock_price
                    self.current_asset += stock_price

                # sell stock
                if stock_share_amount < 0:
                    # if action request more than stock held to sell
                    if - stock_share_amount > self.stock_hold[i]:
                        stock_share_amount = self.stock_hold[i]
                    stock_price = - high_price * stock_share_amount - TRANSACTION_FEE
                    reward += stock_price * 1.5
                    self.current_asset += stock_price

            # calculate interest
            reward -= BANK_INTEREST
            self.current_asset -= BANK_INTEREST

            # check if done
            if self.current_day - self.starting_day >= 998 or self.current_asset < 0:
                self.done = True

            self.current_day += 1
            return np.concatenate(
                (current_price, self.stock_hold, current_history)), reward / 100000, self.done, {}

        else:
            return np.concatenate(
                (self.np_data[self.current_day], self.stock_hold, self.stock_history)), 0, self.done, {}

    def reset(self):
        # reset stock data
        if self.change_stocks:
            self.pd_data = self.sdp.get_data()
            self.np_data = (self.pd_data.drop('Date', axis=1)).to_numpy()

        self.done = False

        # day variables
        self.starting_day = random.randrange(len(self.pd_data) - MAXIMUM_STEPS)
        self.current_day = self.starting_day
        self.stock_history = np.zeros(N_HISTORY * self.stock_quantity * N_STOCK_INFO)

        # stock variables
        self.stock_hold = np.zeros(self.stock_quantity)
        self.current_asset = ASSET

        return np.concatenate((self.np_data[self.current_day], self.stock_hold, self.stock_history), axis=None)

    def render(self, mode='human'):
        print('#################################')
        print('day', self.current_day - self.starting_day + 1)
        for i in range(self.stock_quantity):
            print('stock ', i)
            print('current high:', self.np_data[self.current_day][5 * i + 1])
            print('current low:', self.np_data[self.current_day][5 * i + 2])
            print('hold:', self.stock_hold[i])
        print('asset in hand:', self.current_asset)


class StockTradeDiscrete(gym.Env):

    def __init__(self):
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(low=0, high=np.inf, shape=(1 + N_HISTORY,))
        self.pd_data = pd.read_csv('archive/Data/Stocks/aan.us.txt')
        self.np_data = self.pd_data['Low'].to_numpy()

        self.day = 0
        self.done = False
        self.profit = 0
        self.current_stock_price = self.np_data[self.day]
        self.held_stock_price = 0
        self.held_stock_number = 0
        self.stock_history = np.zeros(N_HISTORY)

    def step(self, action):
        if not self.done:
            reward = 0

            if action == 0:
                self.held_stock_price += self.current_stock_price
                self.held_stock_number += 1

            if action == 1:
                if self.held_stock_number == 0:
                    reward -= 1
                else:
                    reward = (self.held_stock_number * self.current_stock_price - self.held_stock_price)
                    self.profit += reward
                    self.held_stock_price = 0
                    self.held_stock_number = 0

            if self.day >= 999:
                self.done = True

            self.day += 1
            self.current_stock_price = self.np_data[self.day]
            current_history = self.stock_history
            self.stock_history = np.roll(self.stock_history, 1)
            self.stock_history[0] = self.current_stock_price

            if reward > 0:
                reward = 1
            if reward < 0:
                reward = -1

            return np.concatenate(([self.current_stock_price], current_history)), reward, self.done, {}
        else:
            return np.concatenate(([self.current_stock_price], self.stock_history)), 0, self.done, {}

    def reset(self):
        self.day = 0
        self.done = False
        self.profit = 0
        self.current_stock_price = self.np_data[self.day]
        self.held_stock_price = 0
        self.held_stock_number = 0
        self.stock_history = np.zeros(N_HISTORY)
        current_history = self.stock_history
        self.stock_history = np.roll(self.stock_history, 1)
        self.stock_history[0] = self.current_stock_price
        return np.concatenate(([self.current_stock_price], current_history))

    def render(self, mode='human'):
        print('#################################')
        print('day', self.day + 1)
        print('current price:', self.current_stock_price)
        print('hold:', self.held_stock_number)
        print('profit got:', self.profit)

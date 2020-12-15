import random

import gym
import numpy as np
from gym import spaces

from StockDataPreprocessor import StockDataPreprocessor

OBSERVATION_NUMBER = 5
MAXIMUM_STEPS = 1000
ASSET = 1000000
BANK_INTEREST = ASSET * 0.05 / 365
TRANSACTION_FEE = 50
DAILY_TRANSACTION_LIMIT = 100


class StockTrade(gym.Env):

    def __init__(self, path='archive/Data/Stocks', stock_quantity=5, change_stocks=True):
        # define action and observation space
        self.action_space = spaces.Box(low=-1, high=1, shape=(stock_quantity,))
        self.observation_space = spaces.Box(low=0, high=np.inf, shape=(stock_quantity * OBSERVATION_NUMBER,))

        # generate stock data using
        self.sdp = StockDataPreprocessor(path, stock_quantity)

        # store dataframe for data visualization
        self.pd_data = self.sdp.get_data()

        # numpy for presenting states
        self.np_data = (self.pd_data.drop('Date', axis=1)).to_numpy()

        # in game variables
        self.stock_quantity = stock_quantity
        self.change_stocks = change_stocks

        # day variables
        # self.starting_day = random.randrange(len(self.pd_data) - MAXIMUM_STEPS)
        # self.current_day = self.starting_day
        # todo
        self.current_day = 0

        # stock variables
        self.stock_hold = np.zeros(stock_quantity)
        self.current_asset = ASSET

        # game status variable
        self.done = False

    def step(self, action):
        if not self.done:
            state = self.np_data[self.current_day]

            reward = 0

            # calculate reward of stock transaction
            for i in range(self.stock_quantity):
                stock_share_amount = action[i] * DAILY_TRANSACTION_LIMIT
                high_price = state[5 * i + 1]
                low_price = state[5 * i + 2]

                # buy stock
                if stock_share_amount > 0:
                    self.stock_hold[i] += stock_share_amount
                    stock_price = -low_price * stock_share_amount - TRANSACTION_FEE
                    reward += stock_price
                    self.current_asset += stock_price

                # sell stock
                if stock_share_amount < 0:
                    # if action request more than stock held to sell
                    if stock_share_amount > self.stock_hold[i]:
                        stock_share_amount = self.stock_hold[i]
                    stock_price = high_price * stock_share_amount - TRANSACTION_FEE
                    reward += stock_price
                    self.current_asset += stock_price

            # calculate interest
            reward -= BANK_INTEREST
            self.current_asset -= BANK_INTEREST

            # check if done
            # if self.current_day - self.starting_day >= 1000 or self.current_asset < 0:
            if self.current_day >= len(self.pd_data) - 1 or self.current_asset < 0:
                self.done = True

            self.current_day += 1
            return self.np_data[self.current_day], reward, self.done, {}

        else:
            return self.np_data[self.current_day], 0, self.done, {}

    def reset(self):
        # reset stock data
        if self.change_stocks:
            self.pd_data = self.sdp.get_data()
            self.np_data = (self.pd_data.drop('Date', axis=1)).to_numpy()

        self.done = False

        # day variables
        # self.starting_day = random.randrange(len(self.pd_data) - MAXIMUM_STEPS)
        self.current_day = 0

        # stock variables
        self.stock_hold = np.zeros(self.stock_quantity)
        self.current_asset = ASSET

        return self.np_data[self.current_day]

    def render(self, mode='human'):
        pass

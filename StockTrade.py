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
        self.starting_day = random.randrange(len(self.pd_data) - MAXIMUM_STEPS)
        self.current_day = self.starting_day

        # stock variables
        self.stock_hold = np.zeros(stock_quantity)
        self.current_asset = ASSET

    def step(self, action):
        state = self.np_data[self.current_day]




        # increment of day
        self.current_day += 1

    def reset(self):
        # reset stock data
        if self.change_stocks:
            self.pd_data = self.sdp.get_data()
            self.np_data = (self.pd_data.drop('Date', axis=1)).to_numpy()
        # day variables
        self.starting_day = random.randrange(len(self.pd_data) - MAXIMUM_STEPS)
        self.current_day = self.starting_day

        # stock variables
        self.stock_hold = np.zeros(self.stock_quantity)
        self.current_asset = ASSET

    def render(self, mode='human'):
        pass

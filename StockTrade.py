import gym
import numpy as np
from gym import spaces

from StockDataPreprocessor import StockDataPreprocessor

OBSERVATION_NUMBER = 5


class StockTrade(gym.Env):

    def __init__(self, path='archive/Data/Stocks', stock_quantity=5):
        # define action and observation space
        self.action_space = spaces.Box(low=-1, high=1, shape=(stock_quantity,))
        self.observation_space = spaces.Box(low=0, high=np.inf, shape=(stock_quantity * OBSERVATION_NUMBER,))

        # generate stock data using 
        sdp = StockDataPreprocessor()
        self.pd_data = sdp.get_data()
        self.np_data = self.pd_data.drop('Date', axis=1).to_numpy()

    def step(self, action):
        pass

    def reset(self):
        pass

    def render(self, mode='human'):
        pass

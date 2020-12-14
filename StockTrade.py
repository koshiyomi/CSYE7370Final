import gym
from gym import spaces

class StockTrade(gym.Env):

    def __init__(self, stock_quantity, ):
        self.action_space = None
        self.observation_space = None

    def step(self, action):
        pass

    def reset(self):
        pass

    def render(self, mode='human'):
        pass
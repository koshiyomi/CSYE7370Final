from os import listdir
from os.path import isfile, join
from random import sample

import pandas as pd


class StockDataPreProcessor:
    def __init__(self, path='archive/Data/Stocks', stock_quantity=5):
        self.path = path
        self.stock_quantity = stock_quantity

    def get_data(self):
        filenames = [f for f in listdir(self.path) if isfile(join(self.path, f))]
        filenames = sample(filenames, self.stock_quantity)

        stock_data = []
        for filename in filenames:
            stock_data.append(pd.read_csv(self.path + '/' + filename))

        for i in range(len(stock_data)):
            stock_data[i].drop('OpenInt', axis=1, inplace=True)

        joined_data = stock_data[0]
        for i in range(1, len(stock_data)):
            joined_data = pd.merge(joined_data, stock_data[i], how='inner', on='Date')

        return joined_data

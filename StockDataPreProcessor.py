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
            stock_data.append(pd.read_csv(self.path+'/'+filename))

        return filenames


processor = StockDataPreProcessor()
print(len(processor.get_data()))

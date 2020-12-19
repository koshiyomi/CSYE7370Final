from os import listdir
from os.path import isfile, join
from random import sample

import pandas as pd


class StockDataPreprocessor:
    """
    stock data preprocessor for reading, shuffling and output the data into pandas dataframe
    """
    def __init__(self, path='archive/Data/Stocks', stock_quantity=5):
        """
        initializer
        :param path: the path directory which stores all the stock data
        :param stock_quantity: number of stock for the output
        """
        self.path = path
        self.stock_quantity = stock_quantity

    def get_data(self):
        """
        get the dataframe data
        :return: stock data in dataframe
        """
        # read all filename
        filenames = [f for f in listdir(self.path) if isfile(join(self.path, f))]
        # shuffle filenames based on the stock quantity
        filenames = sample(filenames, self.stock_quantity)

        # read all stock data based on the shuffled stock name
        stock_data = []
        for filename in filenames:
            stock_data.append(pd.read_csv(self.path + '/' + filename))

        # drop the unnecessary column
        for i in range(len(stock_data)):
            stock_data[i].drop('OpenInt', axis=1, inplace=True)

        # inner join all stock information into one dataframe
        joined_data = stock_data[0]
        for i in range(1, len(stock_data)):
            joined_data = pd.merge(joined_data, stock_data[i], how='inner', on='Date')

        return joined_data

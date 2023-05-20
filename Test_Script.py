import datetime as dt
import os

import pickle5 as pickle

from collections import Counter
from statistics import mean

#exit()

import matplotlib.pyplot as plt
from matplotlib import style

import numpy as np
import pandas as pd

import pandas_datareader.data as web
import requests
import warnings
warnings.filterwarnings('ignore')

from sklearn import svm, neighbors
from sklearn import model_selection
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import VotingClassifier, RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVC
from sklearn.linear_model import LinearRegression
import bs4 as bs

style.use('ggplot')

def save_stock_symbols():
    rp = requests.get('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')

    obg_sp = bs.BeautifulSoup(rp.text, 'lxml')
    
    stock_table = obg_sp.find('table', {'class': 'wikitable sortable'})

    symbols = []

    for row in stock_table.findAll('tr')[1:]:
        stock_smbl = row.findAll('td')[1].txt
        symbols.append(stock_smbl)
    with open("stocksymbols.pickle", "wb") as i:
        pickle.dump(symbols, i)

        print(symbols)
    return symbols


def fetch_stock_data(stock_loading=False):
    if stock_loading:
        symbol_stocks = save_stock_symbols()
    else:
        with open("stocksymbols.pickle", "rb") as i:
            symbol_stocks = pickle.load(i)

    if not os.path.exists('all_stocks'):
        os.makedirs('all_stocks')

    dt_start = dt.datetime(2010, 1, 1)
    dt_end = dt.datetime(2018, 12, 31)
    for symbol in symbol_stocks[:70]:
        print(symbol)
        if not os.path.exists('all_stocks/{}.csv'.format(symbol)):
            datafr = web.DataReader(symbol, 'yahoo', dt_start, dt_end)

            datafr.to_csv('all_stocks/{}.csv'.format(symbol))
        else:
            print('Exists already {}'.format(symbol))


def stock_compiling():
    with open("stocksymbols.pickle", "rb") as j:
        symbol_stocks = pickle.load(j)

    df_main = pd.DataFrame()
    for count, stock_symbol in enumerate(symbol_stocks[:50]):
        datafr = pd.read_csv('all_stocks/{}.csv'.format(stock_symbol))

        datafr.set_index('Date', inplace = True)

        datafr.rename(columns = {'Adj Close': stock_symbol}, inplace = True)
        datafr.drop(['Open', 'High', 'Low', 'Close', 'Volume'],1  ,inplace = True)

        if df_main.empty:
            df_main = datafr
        else:
            df_main = df_main.join(datafr, how='outer')

    print(df_main.head())

    df_main.to_csv('fifty_Adjusted_CLoses_joined.csv')


def stock_processing(stock):
    day_interval = 7
    datafr = pd.read_csv('fifty_Adjusted_CLoses_joined.csv')

    stocks_ = datafr.columns.values.tolist()

    datafr.fillna(0, inplace = True)
    for i in range(1, day_interval+1):
        datafr['{}_{}day'.format(stock, i)] = (datafr[stock].shift(-i) - datafr[stock]) /datafr[stock]
    datafr.fillna(0, inplace=True) 
    return stocks_, datafr

def decision(*args):
    columns = [i for i in args]
    percentage = 0.02
    for column in columns:
        if column >percentage:
            return 1

        if column < -percentage:
            return -1
    return 0

def feautureset(stock):
    stocks, datafr = stock_processing(stock)

    datafr['{}_stocktrgt'.format(stock)] = list(map(decision, datafr['{}_lday'.format(stock)], datafr['{}_2day'.format(stock)],datafr['{}_3day'.format(stock)],datafr['{}_4day'.format(stock)],
                                                      datafr['{}_5day'.format(stock)], datafr['{}_6day'.format(stock)], datafr['{}_7day'.format(stock)] ))

    values = datafr['{}_stocktrgt'.format(stock)].values.tolist()

    values_of_string = [str(j) for j in values]

    print('Overall Stock Distributuon:', Counter('values_of_string'))

    datafr.fillna(0, inplace=True)
    datafr = datafr.replace([np.inf, -np.inf], np.nan)
    datafr.dropna(inplace = True)

    dataframe_values = datafr[[stock for stock in stocks]].pct_change()
    dataframe_values = dataframe_values.replace([np.inf, -np.inf], 0)
    dataframe_values.fillna(0, inplace=True)
    X = dataframe_values.values
    y=datafr['{}_stocktrgt'.format(stock)].values

    return X, y, datafr

def apply_machine_learning(stock):
    X, y, datafr = feautureset(stock)


    X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size = 0.25)

    algorithm = neighbors.KNeighborsClassifier()


    algorithm.fit(X_train, y_train)
    Evaluation = algorithm.score(X_test, y_test)

    Prediction = algorithm.predict(X_test)
    print('Decision Prediction:', Counter(Prediction))

    return Evaluation


with open("stocksymbols.pickle", "rb") as i:
    stock_symbols = pickle.load(i)

accurs = []
for count, stock in enumerate(stock_symbols[:50]):
    accur = apply_machine_learning(stock)
    print("{} Stock Accuracy: {}. \nAverage accurage:{}".format(stock,accur, mean(accurs)))
    print()
    print()


print('Sum of 70 stocks average accuracy using neighbors.KNeighborsClassifier')
print(sum(accurs)/50)


style.use('ggplot')

df = pd.read_csv("sum_stocks-accuracy.csv")
print(df)

objects = ('K_NEIGHBORS ', 'RANDOM FOREST', 'LINEAR DISCRIMINANT ANALYSIS', 'DECISION TREE', 'PASSIVE AGGRESIVE', 'EXTRA TREES')
y_pos = np.arange(len(objects))
performance = [38.61, 41.83, 41.03, 39.46, 40.25, 40.68]

plt.barh(y_pos, performance, alin = 'center', alpha = 0.5)
plt.yticks(y_pos, objects)
plt.xlabel('Accuracy')
plt.ttle('Algorithm Used')

plt.show()
df.set_index()[['KNN', 'RFC', 'SVC',]].plot.bar()
plt.show()

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

from sklearn import svm, neighbours
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.emsemble import VotingClassifier, RandomForestClassifier
from sklearn.linear_model import LinearRegression


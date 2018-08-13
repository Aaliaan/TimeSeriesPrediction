from math import sqrt
from numpy import concatenate
from matplotlib import pyplot
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
import plotly.offline as py
import plotly.graph_objs as go
import numpy as np
import seaborn as sns
import urllib.request, json
from pandas.io.json import json_normalize
from pandas.io.json import json_normalize
py.init_notebook_mode(connected=True)

def grab_data_from_api():
    with urllib.request.urlopen("https://min-api.cryptocompare.com/data/histominute?fsym=BTC&tsym=USD&limit=500000") as url:
        d = json.load(url)
        dataframe = json_normalize(d['Data'])
    return dataframe


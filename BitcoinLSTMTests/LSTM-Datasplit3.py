from math import sqrt
from numpy import concatenate
from matplotlib import pyplot
import pandas as pd
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
import plotly.offline as py
import plotly.graph_objs as go
import numpy as np
import seaborn as sns
py.init_notebook_mode(connected=True)
data = pd.read_csv('2014_BTCUSD_1m.csv', index_col='Timestamp').dropna(how='any')

# print fixed data

data[data['Weighted_Price']<1]=None
btc_trace = go.Scatter(x=data.index, y=data['Weighted_Price'], name='PRICE')
py.plot([btc_trace])

# check if there are NaN
print(data.isnull().sum())

data = data.dropna(how='any')
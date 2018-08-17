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
    print('Grabbed data')
    print('Initialised dataframe')
    return dataframe


def remove_null_values(data):
    # puts time as the dataframe index
    data.set_index('time', inplace=True)
    print('Indexed columns')
    data[data['close'] < 1] = None
    print('Locating bad values')
    data.isnull().sum()
    data = data.dropna(how='any')
    print('Dropped bad values')
    return data


def reshape_values(data):
    values = data['close'].values.reshape(-1, 1)
    values = values.astype('float64')
    return values


def apply_minmaxscaler(values, scaler):
    scaled = scaler.fit_transform(values)
    print('Normalised data')
    return scaled


def create_train(scaled):
    train_size = int(len(scaled) * 0.7)
    train = scaled[0:train_size, :]
    print('Length of train: %.3f' % len(train))
    return train


def create_test(scaled):
    train_size = int(len(scaled) * 0.7)
    test_size = len(scaled) - train_size
    test = scaled[train_size:len(scaled), :]
    print('Length of test: %.3f' % len(test))
    return test


def create_dataset(dataset, look_back=1):
    dataX, dataY = [], []
    for i in range(len(dataset) - look_back):
        a = dataset[i:(i + look_back), 0]
        dataX.append(a)
        dataY.append(dataset[i + look_back, 0])
    print(len(dataY))
    print('Look back dataset method complete')
    return np.array(dataX), np.array(dataY)


def simple_lstm(testX, testY, trainX, trainY):
    model = Sequential()
    model.add(LSTM(100, input_shape=(trainX.shape[1], trainX.shape[2])))
    model.add(Dense(1))
    model.compile(loss='mae', optimizer='adam')
    history = model.fit(trainX, trainY, epochs=300, batch_size=100, validation_data=(testX, testY), verbose=0, shuffle=False)

    def plot1(history):
        pyplot.plot(history.history['loss'], label='train')
        pyplot.plot(history.history['val_loss'], label='test')
        pyplot.legend()
        pyplot.show()
        return 0


    def plot2(history):
        yhat = model.predict(testX)
        pyplot.plot(yhat, label='predict')
        pyplot.plot(testY, label='true')
        pyplot.legend()
        pyplot.show()
        return yhat

    plot1(history)
    yhat = plot2(history)
    return yhat


def inverse_values(scaler, yhat, testY):
    yhat_inverse = scaler.inverse_transform(yhat.reshape(-1, 1))
    testY_inverse = scaler.inverse_transform(testY.reshape(-1, 1))
    return yhat_inverse, testY_inverse


def plot_inverse_graph(inversehat, inversetest):
    pyplot.plot(inversehat, label='predict')
    pyplot.plot(inversetest, label='actual', alpha=0.7)
    pyplot.legend()
    pyplot.show()
    return 0


def calculate_rmse(dataY, yhat):
    rmse = sqrt(mean_squared_error(dataY, yhat))
    print('Test RMSE: %.3f' % rmse)
    return 0


def reshape_visualise(data, testX, testY_inverse, yhat_inverse):
    '''
    reshapes values
    outputs with plotly object
    :returns testY and yhat reshaped
    '''
    predictDates = data.tail(len(testX)).index
    testY_reshape = testY_inverse.reshape(len(testY_inverse))
    yhat_reshape = yhat_inverse.reshape(len(yhat_inverse))
    actual_chart = go.Scatter(x=predictDates, y=testY_reshape, name='Actual')
    predict_chart = go.Scatter(x=predictDates, y=yhat_reshape, name='Predicted')
    py.plot([predict_chart, actual_chart])
    return testY_reshape, yhat_reshape


def plot_heatmap(data):
    sns.heatmap(data.corr(), annot=True, cmap='RdYlGn', linewidths=0.1, vmin=0, edgecolor='White')
    return 0


def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    n_vars = 1 if type(data) is list else data.shape[1]
    df = pd.DataFrame(data)
    cols, names = list(), list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j+n_in, i)) for j in range(n_vars)]
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j+n_in)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j+n_in, i)) for j in range(n_vars)]
    # put it all together
    agg = pd.concat(cols, axis=1)
    agg.columns = names
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg


if __name__ == "__main__":
    data = grab_data_from_api()
    data = remove_null_values(data)
    values = reshape_values(data)
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled = apply_minmaxscaler(values, scaler)
    train = create_train(scaled)
    test = create_test(scaled)

    look_back = 1
    trainX, trainY = create_dataset(train, look_back)
    testX, testY = create_dataset(train, look_back)
    trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
    testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))

    model = Sequential()
    print('TrainX shape:')
    print(trainX.shape[1])
    model.add(LSTM(250, input_shape=(trainX.shape[1], trainX.shape[2])))

    model.add(Dense(1))
    #TODO: Try mean squared error, try logcosh error fuctions - two at extreme ends.
    #Require linear layer on top of hidden... LSTM output = (cell_state, hidden_state)
    model.compile(loss='mae', optimizer='adam')
    history = model.fit(trainX, trainY, epochs=300, batch_size=100, validation_data=(testX, testY), verbose=0,
                        shuffle=False)

    yhat = model.predict(testX)
    pyplot.plot(yhat, label='predict')
    pyplot.plot(testY, label='true')
    pyplot.legend()
    pyplot.show()

    yhat_inverse, testY_inverse = inverse_values(scaler, yhat, testY)
    calculate_rmse(testY, yhat)
    plot_inverse_graph(yhat_inverse, testY)
    testY_reshape, yhat_reshape = reshape_visualise(data, testX, testY_inverse, yhat_inverse)
    calculate_rmse(testY_reshape, yhat_reshape)
    plot_heatmap(data)








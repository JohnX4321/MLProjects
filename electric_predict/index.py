from numpy import nan, isnan
from pandas import read_csv, to_numeric
from math import sqrt
from numpy import split, array

from sklearn.metrics import mean_squared_error
from matplotlib import pyplot

from statsmodels.tsa.arima_model import ARIMA


def evaluate_forecast(actual, predicted):
    scores = list()
    for i in range(actual.shape[1]):
        mse = mean_squared_error(actual[:, i], predicted[:, i])
        rmse = sqrt(mse)
        scores.append(rmse)
    s = 0
    for row in range(actual.shape[0]):
        for col in range(actual.shape[1]):
            s += (actual[row, col] - predicted[row, col]) ** 2
    score = sqrt(s / actual.shape[0] * actual.shape[1])
    return score, scores


def split_dataset(data):
    train, test = data[1:-328], data[-328:-6]
    train = array(split(train, len(train) / 7))
    test = array(split(test, len(test) / 7))
    return train, test


dataset = read_csv('household_power_consumption_days.csv', header=0, infer_datetime_format=True,
                   parse_dates=['datetime'], index_col=['datetime'])
train, test = split_dataset(dataset.values)

print(train.shape)
print(train[0, 0, 0], train[-1, -1, 0])

print(test.shape)
print(test[0, 0, 0], test[-1, -1, 0])


def evaluate_model(model_func, train, test):
    history = [x for x in train]
    predictions = list()
    for i in range(len(test)):
        yhat_seq = model_func(history)
        predictions.append(yhat_seq)
        history.append(test[i, :])
    predictions = array(predictions)
    score, scores = evaluate_forecast(test[:, :, 0], predictions)
    return score, scores


def summarize_scores(name, score, scores):
    s_scores = ', '.join(['%.1f' % s for s in scores])
    print('%s: [%.3f] %s' % (name, score, s_scores))


def to_series(data):
    series = [week[:, 0] for week in data]
    series = array(series).flatten()
    return series


def arima_forecast(history):
    series = to_series(history)
    model = ARIMA(series, order=(7, 0, 0))
    model_fit = model.fit(disp=False)
    yhat = model_fit.predict(len(series), len(series) + 6)
    return yhat


def daily_persistence(history):
    last_week = history[-1]
    value = last_week[-1, 0]
    forecast = [value for _ in range(7)]
    return forecast


def weekly_persistence(history):
    last_week = history[-1]
    return last_week[:, 0]


def week_one_year_ago_persistence(history):
    # get the data for the prior week
    last_week = history[-52]
    return last_week[:, 0]


dataset = read_csv('household_power_consumption_days.csv', header=0, infer_datetime_format=True,
                   parse_dates=['datetime'], index_col=['datetime'])
# split into train and test
train, test = split_dataset(dataset.values)

models = dict()
models['arima'] = arima_forecast
# models['daily'] = daily_persistence
# models['weekly'] = weekly_persistence
# models['week-oya'] = week_one_year_ago_persistence
# evaluate each model
days = ['sun', 'mon', 'tue', 'wed', 'thr', 'fri', 'sat']
for name, func in models.items():
    # evaluate and get scores
    score, scores = evaluate_model(func, train, test)
    # summarize scores
    summarize_scores(name, score, scores)
    # plot scores
    pyplot.plot(days, scores, marker='o', label=name)
# show plot
pyplot.legend()
pyplot.show()

import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')
plt.style.use('fivethirtyeight')
from pylab import rcParams

rcParams['figure.figsize'] = 10, 7
df = pd.read_csv('Electric_Production.csv')
df.columns = ['Date', 'Consumption']
df = df.dropna()
df['Date'] = pd.to_datetime(df['Date'])
df.set_index('Date', inplace=True)
df.head()

plt.xlabel('Date')
plt.ylabel('Consumption')
plt.title('Production Graph')
plt.plot(df)
df.plot(style='k.')
plt.show()

from statsmodels.tsa.seasonal import seasonal_decompose

res = seasonal_decompose(df, model='multiplicative')
res.plot()
plt.show()

from statsmodels.tsa.stattools import adfuller


def stationarity(timeseries):
    rolmean = timeseries.rolling(12).mean()
    rolstd = timeseries.rolling(12).std()
    plt.plot(timeseries, color='blue', label='Original')
    plt.plot(rolmean, color='red', label='Rolling Nean')
    plt.plot(rolstd, color='black', label='Rolling Std')
    plt.legend(loc='best')
    plt.title('Rolling Mean and Standard Deviation')
    plt.show(block=False)
    print("Results of dickey fuller test")
    adft = adfuller(timeseries['Consumption'], autolag='AIC')
    output = pd.Series(adft[0:4],
                       index=['Test Statistics', 'p-value', 'No. of lags used', 'Number of observations used'])
    for key, values in adft[4].items():
        output['critical value (%s)' % key] = values
    print(output)


stationarity(df)

df_log = np.log(df)
moving_avg = df_log.rolling(12).mean()
std_dev = df_log.rolling(12).std()
plt.plot(df_log)
plt.plot(moving_avg, color="red")
plt.plot(std_dev, color="black")
plt.show()

df_log_moving_avg_diff = df_log - moving_avg
df_log_moving_avg_diff.dropna(inplace=True)

stationarity(df_log_moving_avg_diff)

weighted_avg = df_log.ewm(halflife=12, min_periods=0, adjust=True).mean()
logScale = df_log - weighted_avg
from pylab import rcParams

rcParams['figure.figsize'] = 10, 6
stationarity(logScale)

df_log_diff = df_log - df_log.shift()
plt.title("Shifted timeseries")
plt.xlabel("Date")
plt.ylabel("Consumption")
plt.plot(df_log_diff)
# Let us test the stationarity of our resultant series
df_log_diff.dropna(inplace=True)
stationarity(df_log_diff)

from chart_studio.plotly import plot_mpl
from statsmodels.tsa.seasonal import seasonal_decompose

result = seasonal_decompose(df_log, model='additive', freq=12)
result.plot()
plt.show()
trend = result.trend
trend.dropna(inplace=True)
seasonality = result.seasonal
seasonality.dropna(inplace=True)
residual = result.resid
residual.dropna(inplace=True)
stationarity(residual)

from statsmodels.tsa.stattools import acf, pacf

# we use d value here(data_log_shift)
acf = acf(df_log_diff, nlags=15)
pacf = pacf(df_log_diff, nlags=15, method='ols')
# plot PACF
plt.subplot(121)
plt.plot(acf)
plt.axhline(y=0, linestyle='-', color='blue')
plt.axhline(y=-1.96 / np.sqrt(len(df_log_diff)), linestyle='--', color='black')
plt.axhline(y=1.96 / np.sqrt(len(df_log_diff)), linestyle='--', color='black')
plt.title('Auto corellation function')
plt.tight_layout()
# plot ACF
plt.subplot(122)
plt.plot(pacf)
plt.axhline(y=0, linestyle='-', color='blue')
plt.axhline(y=-1.96 / np.sqrt(len(df_log_diff)), linestyle='--', color='black')
plt.axhline(y=1.96 / np.sqrt(len(df_log_diff)), linestyle='--', color='black')
plt.title('Partially auto corellation function')
plt.tight_layout()

from statsmodels.tsa.arima_model import ARIMA

model = ARIMA(df_log, order=(3, 1, 3))
result_AR = model.fit(disp=0)
# plt.plot(df_log_diff)
# plt.plot(result_AR.fittedvalues, color='red')
# plt.title("sum of squares of residuals")
# print('RSS : %f' %sum((result_AR.fittedvalues-df_log_diff["Consumption"])**2))
result_AR.plot_predict(1, 800)
x = result_AR.forecast(steps=200)

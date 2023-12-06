import datetime as dt
import matplotlib.pyplot as plt
from matplotlib import style
from mpl_finance import candlestick_ohlc

import matplotlib.dates as mdates
import numpy as np
import seaborn as sns
from sklearn import metrics
from sklearn.model_selection import train_test_split


import pandas as pd
import pandas_datareader.data as web
import yfinance as yf

style.use('ggplot')

#Need to provide a start date and end date for which the stock data has to be retreived for Tesla from yahoo finance api

start = dt.datetime(2000,1,1)
end = dt.datetime(2022,12,31)

stock_symbol = 'TSLA'

# Download stock data from Yahoo Finance
df = yf.download(stock_symbol, start=start, end=end)
df.to_csv('Tesla.csv')




# Tesla closing prices plotted
df = pd.read_csv('Tesla.csv',parse_dates=True,index_col=0)
print(df.head())

df['Adj Close'].plot()
plt.show()



#Since there are too many datapoints,we will resample the data for 10 days i.e convert 10 day data into 1 point to make
#it easier for us to understand the data
#df_ohlc - ohlc stands for Open High Low Close.In Financial Markets,the stock prices are generally represented with
#the help of candlestick charts,the candlestick chart represent Open,High Low and Close
#we use the resample.ohlc method inbuilt in pandas to convert the data into 10 day data
#For Volume in place of taking the ohlc or mean,we take the sum over 10 days
df['100ma'] = df['Adj Close'].rolling(window=100).mean()


print(df.head())
print(df.tail())

#Candlestick representation:
df_ohlc = df['Adj Close'].resample('10D').ohlc()
df_volume = df['Volume'].resample('10D').sum()
df_ohlc.reset_index(inplace=True)
df_ohlc['Date'] = df_ohlc['Date'].map(mdates.date2num)
print(df_ohlc.head())


# Plotting candlestick chart
ax1 = plt.subplot2grid((6,1) ,(0,0), rowspan=5, colspan=1)
ax2 = plt.subplot2grid((6,1) ,(5,0), rowspan=1, colspan=1, sharex=ax1)
ax1.xaxis_date()
candlestick_ohlc(ax1, df_ohlc.values, width=2, colorup='g')
ax2.fill_between(df_volume.index.map(mdates.date2num), df_volume.values, 0)
plt.show()

#Tesla correlation with 25 other companies
def tesla_corr_heatmap():
    df2 = pd.read_csv('S&P500_joined_closes.csv').fillna(value = 0)
    df1 = pd.read_csv('Tesla.csv').fillna(value = 0)

    # print(df2.describe)
    print(df1.head())
    print(df2.head())

    correlation = df1["Adj Close"].corr(df2["MMM"])

    print(df1["Adj Close"].isna().any())
    # for column in df2:
    #     print(df2[column])
    corr_arr = []
    tickers = []
    for column in df2.columns[1:]:
        print(df2[column].isna().any())
        corr_arr.append(df1["Adj Close"].corr(df2[column]))
        tickers.append(column)
    
    print(corr_arr)
    print(tickers)

    plt.figure(figsize=(12, 6)) 
    plt.bar(tickers,corr_arr)
    plt.xlabel('Stock Codes')
    plt.ylabel('Correlation')
    plt.title('Correlation of Tesla with S&P 500 Stocks')
    plt.xticks(rotation=90)
    plt.show()

    print(correlation)


tesla_corr_heatmap()


#Visualising part is complete

#Now analyze the results of all the data that we have

#We can start with predicting the stock prices with linear regression

print(df.head(10))

x = df.index
y = df['Adj Close']

def df_plot(data, x, y, title="", xlabel='Date', ylabel='Value', dpi=100):
    plt.figure(figsize=(16,5), dpi=dpi)
    plt.plot(x, y, color='tab:red')
    plt.gca().set(title=title, xlabel=xlabel, ylabel=ylabel)
    plt.show()


stock_name= "TESLA"
title = (stock_name,"History stock performance till date")

df_plot(df , x , y , title=title,xlabel='Date', ylabel='Value',dpi=100)

df.reset_index(inplace=True) # to reset index and convert it to column
df.drop(columns=['Close'], inplace=True)



print(df.describe())
print(len(df))

#SIMPLE MOVING AVERAGE
window = 50
df['SMA'] = df['Adj Close'].rolling(window=window).mean()

# Plot the stock prices and SMA
plt.figure(figsize=(16,5), dpi=100)
plt.plot(df['Date'], df['Adj Close'], color='tab:red', label='Adj Close', alpha=0.7)
plt.plot(df['Date'], df['SMA'], color='tab:blue', label=f'{window}-Day SMA', alpha=0.7)
plt.gca().set(title='Tesla Stock Price with SMA', xlabel='Date', ylabel='Price')
plt.legend()
plt.show()

#EXPONENTIAL MOVING AVERAGE
period = 20
df['EMA'] = df['Adj Close'].ewm(span=period, adjust=False).mean()

# Plot the stock prices and EMA
plt.figure(figsize=(16, 5), dpi=100)
plt.plot(df['Date'], df['Adj Close'], color='tab:red', label='Adj Close', alpha=0.7)
plt.plot(df['Date'], df['EMA'], color='tab:blue', label=f'{period}-Day EMA', alpha=0.7)
plt.gca().set(title='Tesla Stock Price with EMA', xlabel='Date', ylabel='Price')
plt.legend()
plt.show()


x = df[['Open', 'High','Low', 'Volume']]
y = df['Adj Close']
# Linear regression Model for stock prediction 
train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.15 , shuffle=False,random_state = 0)
print(train_x.shape )
print(test_x.shape)
print(train_y.shape)
print(test_y.shape)


from sklearn.linear_model import LinearRegression
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.metrics import mean_squared_error, r2_score

regression = LinearRegression()
regression.fit(train_x, train_y)
print("regression coefficient",regression.coef_)
print("regression intercept",regression.intercept_)

#Here we will compute the coefficient of determination denoted by R², which takes values between 0 and 1, the higher the value R² the more successful the linear regression is at explaining the variation of Y values, in our case the Y values represent the close stock prices of the subjected company. The below is the math behind The coefficient of determination R²

# the coefficient of determination R² 
regression_confidence = regression.score(test_x, test_y)
print("linear regression confidence: ", regression_confidence)


predicted=regression.predict(test_x)
print(test_x.head())

print(predicted.shape)


dfr=pd.DataFrame({'Actual_Price':test_y, 'Predicted_Price':predicted})
#Adding actual dates to the newly created dataframe to make it easier to understand
print(dfr.head())
num_data_points = len(dfr)
print(num_data_points)
start_date = '2021-01-29'  # Replace with your desired start date
date_range = pd.date_range(start=start_date, periods=num_data_points)
dfr['Date'] = date_range


print(dfr.head(10))
print(dfr.describe())
dfr.to_csv('ActualvPredicted_prices.csv')

#Accuracy of the model
x2 = dfr.Actual_Price.mean()
y2 = dfr.Predicted_Price.mean()
Accuracy1 = x2/y2*100
print("The accuracy of the model is " , Accuracy1)

#Scatterplot
plt.scatter(dfr.Actual_Price, dfr.Predicted_Price,  color='Darkblue')
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.show()



# #Prediction Chart- Tesla
x = dfr['Date']
y_actual = dfr['Actual_Price']
y_predicted = dfr['Predicted_Price']
plt.plot(x, y_actual, color='black', label='Actual Price')
plt.plot(x, y_predicted, color='red', label='Predicted Price')
plt.title("Tesla Prediction Chart")
plt.xlabel("Date")
plt.ylabel("Price")
plt.legend(fontsize="x-large")
plt.xticks(rotation=45)
plt.show()

mse = mean_squared_error(test_y, predicted)
print(f'Mean Squared Error: {mse}')

# R-squared
r2 = r2_score(test_y, predicted)
print(f'R-squared: {r2}')
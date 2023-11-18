import bs4 as bs
import pickle
import requests
import datetime as dt
import os
import pandas as pd
import pandas_datareader.data as web
import yfinance as yf
import matplotlib.pyplot as plt
from matplotlib import style
import numpy as np


style.use('ggplot')

def save_sp500_tickers():
    #Using 'requests' library to fetch the HTML content of the Wikipedia page
    resp = requests.get('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
    #Using beautifulsoup to parse and extract information from HTML
    soup = bs.BeautifulSoup(resp.text,"lxml")
    #We will look for a table element in the html file with the class - 'wikitable sortable'
    table  = soup.find('table', {'class': 'wikitable sortable'})
    #Creating a tickers array to store all the tickers in the list.So tickers are basically the short names of the companies that are used in S&P 500 index
    
    tickers = []
    
    #we will now iterate through all the rows in the extracted table and append the values in the first column i.e. column 0 to our tickers array.
    for row in table.findAll('tr')[1:]:
        ticker = row.findAll('td')[0].text
        tickers.append(ticker)
        
    #After the loop, we open a binary file named "sp500tickers.pickle" in write mode ("wb") using a with statement.The whole point of this below code is to avoid requesting the data and scrape it every time we run the code.    
    with open("sp500tickers.pickle","wb")  as f:
        pickle.dump(tickers, f)
        
    # print(tickers)    
    return tickers

# save_sp500_tickers()

def get_data_from_yahoo(reload_sp500=False):
    if reload_sp500:
        tickers = save_sp500_tickers()
    else:
        with open("sp500tickers.pickle","rb")  as f:
            tickers = pickle.load(f)
              
    if not os.path.exists('stock_dfs'):
        os.makedirs('stock_dfs')     
    start = dt.datetime(2000,1,1)
    end = dt.datetime(2022,12,31)
    for ticker in tickers[:25]:
        ticker = ticker.strip()
        print(ticker)
        if not os.path.exists('stock_dfs/{}.csv'.format(ticker)):
            df = yf.download(ticker, start=start, end=end)
            df.to_csv(f'stock_dfs/{ticker}.csv')
        else:
            print(f'Already have {ticker}')


get_data_from_yahoo()


def compile_data():
        with open("sp500tickers.pickle","rb")  as f:
            tickers = pickle.load(f)
            
        main_df  = pd.DataFrame()
        
        for count,ticker in enumerate(tickers[:25]):
            ticker = ticker.strip()
            df = pd.read_csv(f'stock_dfs/{ticker}.csv')
            df.set_index('Date',inplace=True)
            
            df.rename(columns= {'Adj Close' : ticker},inplace=True)
            
            df.drop(['Open','High','Low','Close','Volume'], 1 ,inplace=True)
            
            
            if main_df.empty:
                main_df = df
            else:
                main_df = main_df.join(df,how='outer')
                
            if count % 10 == 0:
                print(count)
                
            print(main_df.head())
            main_df.to_csv('S&P500_joined_closes.csv')
        
        
# compile_data()


def visualize_data():
    #Below line reads a CSV file named 'S&P500_joined_closes.csv' using the Pandas library and stores the data in a DataFrame called df
    df = pd.read_csv('S&P500_joined_closes.csv')
    df_corr = df.corr()
    #This line calculates the correlation matrix of the DataFrame df, which represents how the stock prices of different companies are related to each other. The resulting correlation matrix is stored in the DataFrame df_corr
    print(df_corr.head())
    data = df_corr.values
    # This line extracts the values from the df_corr DataFrame and stores them in a NumPy array called data. This array will be used to create the heatmap.
    fig = plt.figure()
    #with this,we create a new Matplotlib figure that will contain the heatmap.
    ax = fig.add_subplot(1,1,1)
    #With this we create a subplot with a subplot.Since we only have 1 subplot we use the argument 111

    heatmap = ax.pcolor(data, cmap= plt.cm.RdYlGn)
    #Now we create a heatmap created with ax subplot ranging from Red to Green.Red stands for negative corr.,yellow for no corr. and green for positive corr.
    
    #We now create a colorbar at the side indicating the colors and the values
    fig.colorbar(heatmap)
    
    #What the below 2 lines do is , place the xticks and yticks values by placing them at the center of each cell,minor=False indicate that the ticks are not minor
    #Ticks are basically the Stock codes for each stock.We want to label them correctly,so we center them as per our data points
    ax.set_xticks(np.arange(data.shape[0]) + 0.5, minor=False)
    ax.set_yticks(np.arange(data.shape[1]) + 0.5, minor=False)
    

    ax.invert_yaxis()
    #We invert y-axis so that its easier to read the heatmap
    ax.xaxis.tick_top()
    #Then we move the x-axis ticks to the top.Totally optional.
    
    columns_labels = df_corr.columns
    row_labels = df_corr.index
    #We then create the rows and columns to be used in our heatmap
    #In our case,we use df_corr.columns as the column labels and df_corr.index as the row labels
    
    ax.set_xticklabels(columns_labels)
    ax.set_yticklabels(row_labels)
    #Now we set the rows and columns in the heatmap
    
    plt.xticks(rotation=90)
    #rotate the labels on x-axis which is at the top by 90 degree,just so that we see all the stock codes clearly.
    
    
    heatmap.set_clim(-1,1)
    #Setting the colorscale in the heatmap from -1 to 1.Just how correalations work!

    plt.tight_layout()
    plt.show()
        

visualize_data()
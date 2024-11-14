import yfinance as yf
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Download wheat and corn futures data from Yahoo Finance
data_wheat = yf.download("ZW=F", start="2013-01-01", end="2023-05-01")
data_wheat['wheat_return'] = data_wheat['Adj Close'].pct_change()

data_corn = yf.download("ZC=F", start="2013-01-01", end="2023-05-01")
data_corn['corn_return'] = data_corn['Adj Close'].pct_change()


# Merge the two datasets on the date index
data = data_wheat[['wheat_return']].merge(data_corn[['corn_return']], left_index=True, right_index=True)

# Define states based on returns of both wheat and corn
conditions = [
    (data['wheat_return'] >= 0) & (data['corn_return'] >= 0),
    (data['wheat_return'] >= 0) & (data['corn_return'] < 0),
    (data['wheat_return'] < 0) & (data['corn_return'] >= 0),
    (data['wheat_return'] < 0) & (data['corn_return'] < 0)
]
states = ['both_up', 'wheat_up_corn_down', 'wheat_down_corn_up', 'both_down']
data['state'] = np.select(conditions, states)


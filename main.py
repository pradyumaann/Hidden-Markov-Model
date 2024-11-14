import numpy as np
from hmmlearn import hmm
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf

# Download wheat futures data from Yahoo Finance
data = yf.download("ZW=F", start="2018-01-01", end="2023-05-01")
data['returns'] = data['Adj Close'].pct_change()

# Feature engineering
data['sma_20'] = data['Adj Close'].rolling(window=20).mean()
data['rsi'] = data.apply(lambda row: talib.RSI(data['Adj Close'], timeperiod=14)[row.name], axis=1)
data['macd'], _, _ = talib.MACD(data['Adj Close'], fastperiod=12, slowperiod=26, signalperiod=9)

# Prepare data for HMM
X = data[['returns', 'sma_20', 'rsi', 'macd']].values

# Define and train HMM
model = hmm.GaussianHMM(n_components=3, covariance_type="diag", n_iter=100)
model.fit(X)

# Get hidden states and calculate transition matrix
hidden_states = model.predict(X)
transition_matrix = model.transmat_

# Develop trading strategy
positions = []
state = hidden_states[0]

for i in range(1, len(hidden_states)):
    if hidden_states[i] != state:
        # Check transition probabilities to determine trade signal
        if transition_matrix[state, hidden_states[i]] > 0.6:
            if hidden_states[i] == 0:
                positions.append('long')
            elif hidden_states[i] == 1:
                positions.append('short')
            else:
                positions.append('hold')
        else:
            positions.append('hold')
        state = hidden_states[i]
    else:
        positions.append('hold')

data['position'] = positions
data['pnl'] = data['position'].map({'long': data['returns'], 'short': -data['returns'], 'hold': 0}) * data['Adj Close'].shift(1)
data['strategy_return'] = data['pnl'].cumsum()

# Evaluate and visualize results
print(f"Final strategy return: {data['strategy_return'].iloc[-1]:.2%}")
data['strategy_return'].plot()
plt.show()
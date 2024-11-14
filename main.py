import yfinance as yf
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Download wheat and corn futures data from Yahoo Finance
data_wheat = yf.download("ZW=F", start="2020-01-01", end="2024-10-01")
data_wheat['wheat_return'] = data_wheat['Adj Close'].pct_change()

data_corn = yf.download("ZC=F", start="2020-01-01", end="2024-10-01")
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

# Shift states to calculate second-order transitions
data['prev_state'] = data['state'].shift(1)
data['two_prev_state'] = data['state'].shift(2)

# Remove rows with NaN values caused by shifting
data.dropna(inplace=True)

# Define all possible second-order transitions for a Markov process with wheat and corn states
second_order_transitions = [(s1, s2, s3) for s1 in states for s2 in states for s3 in states]

# Calculate second-order transition probabilities
transition_counts = {}
for (s1, s2, s3) in second_order_transitions:
    count = len(data[(data['two_prev_state'] == s1) & (data['prev_state'] == s2) & (data['state'] == s3)])
    total_count = len(data[(data['two_prev_state'] == s1) & (data['prev_state'] == s2)])
    probability = count / total_count if total_count > 0 else 0
    transition_counts[(s1, s2, s3)] = probability

# Create a DataFrame for the second-order transition matrix
transition_matrix = pd.DataFrame(
    {
        (s1, s2): [transition_counts[(s1, s2, s3)] for s3 in states]
        for s1 in states for s2 in states
    },
    index=states
)

print("Second-Order Transition Matrix with Correlated Commodity:")
print(transition_matrix)

# Visualize the transition matrix as a heatmap
plt.figure(figsize=(14, 10))
sns.heatmap(transition_matrix, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Second-Order Transition Probability Matrix with Wheat and Corn States")
plt.xlabel("(Previous State 1, Previous State 2)")
plt.ylabel("Current State")
plt.show()

# Contrarian Trading Signals based on transition probabilities
def generate_trading_signal(row, threshold=0.5):
    """Generate contrarian trading signals based on transition probabilities."""
    prev1, prev2 = row['prev_state'], row['two_prev_state']
    if (prev1, prev2) in transition_matrix.columns:
        # Extract transition probabilities for the current previous states
        prob_up = transition_matrix[(prev1, prev2)]['both_up']
        prob_down = transition_matrix[(prev1, prev2)]['both_down']
        
        # Contrarian trading signal based on low probability
        if prob_up < threshold:
            return "Buy"  # Low probability of uptrend, so contrarian Buy
        elif prob_down < threshold:
            return "Sell"  # Low probability of downtrend, so contrarian Sell
        else:
            return "Hold"  # No strong contrarian signal
    else:
        return "Hold"  # Default to hold if no data

# Apply the trading signal function
data['signal'] = data.apply(generate_trading_signal, axis=1)

# Calculate daily returns based on signals
data['strategy_return'] = np.where(data['signal'] == 'Buy', data['wheat_return'],
                                   np.where(data['signal'] == 'Sell', -data['wheat_return'], 0))

# Calculate cumulative returns
data['cumulative_strategy_return'] = (1 + data['strategy_return']).cumprod()
data['cumulative_wheat_return'] = (1 + data['wheat_return']).cumprod()

# Final cumulative return values
final_strategy_return = data['cumulative_strategy_return'].iloc[-1]
final_wheat_return = data['cumulative_wheat_return'].iloc[-1]

# Print the final cumulative returns
print(f"Final Cumulative Return for Contrarian Strategy: {final_strategy_return:.2f}")
print(f"Final Cumulative Return for Buy & Hold Strategy: {final_wheat_return:.2f}")

# Display a portion of the data with trading signals and returns
print(data[['wheat_return', 'corn_return', 'state', 'prev_state', 'two_prev_state', 'signal', 
            'strategy_return', 'cumulative_strategy_return', 'cumulative_wheat_return']].tail(20))

# Plot cumulative returns
plt.figure(figsize=(12, 6))
plt.plot(data.index, data['cumulative_strategy_return'], label='Strategy Cumulative Return', color='blue')
plt.plot(data.index, data['cumulative_wheat_return'], label='Wheat Cumulative Return', color='orange')
plt.xlabel("Date")
plt.ylabel("Cumulative Return")
plt.title("Cumulative Return of Contrarian Strategy vs. Wheat")
plt.legend()
plt.show()

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
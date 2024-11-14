
## Hidden-Markov Trading Model for Wheat and Corn Futures

Using a second-order Markov model, this project implements a contrarian trading strategy for wheat and corn futures. The model generates trading signals by identifying low-probability events in historical price transitions, aiming to take advantage of potential reversals in these commodities markets.

## Overview

The contrarian strategy uses historical daily returns for wheat and corn to model market states and identify unusual state transitions. By focusing on low-probability transitions, the model generates buy and sell signals that may capitalize on market reversals. The model's performance is compared to a buy-and-hold strategy for wheat futures.

## Methodology

1. **Data Collection**: The model retrieves historical data for wheat and corn futures from Yahoo Finance.
2. **State Definition**: Market states are defined based on the daily returns of both wheat and corn, resulting in four states:
   - Both up
   - Wheat up, corn down
   - Wheat down, corn up
   - Both down
3. **Second-Order Markov Model**: A second-order Markov transition matrix is created to calculate the probability of moving from two consecutive previous states to a current state.
4. **Contrarian Signals**: For transitions with probabilities below a specified threshold, the model generates contrarian trading signals:
   - "Buy" if the low-probability event indicates an uptrend.
   - "Sell" if it indicates a downtrend.
5. **Performance Evaluation**: Cumulative returns from the contrarian strategy are compared to a simple buy-and-hold strategy for wheat futures.

## Setup and Usage

1. **Dependencies**: Install the required libraries:
   ```bash
   pip install yfinance pandas numpy seaborn matplotlib
   ```

2. **Running the Model**: Run the Python script to generate trading signals, calculate cumulative returns, and visualize the performance.

3. **Outputs**:
   - The second-order transition probability matrix for wheat and corn states.
   - Trading signals based on low-probability transitions.
   - Cumulative returns for the contrarian strategy vs. the buy-and-hold approach.

## Files

- `main.py`: Main script for data retrieval, transition matrix calculation, trading signal generation, and performance visualization.
- `README.md`: This file, providing an overview of the project.




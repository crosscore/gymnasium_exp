"""
Configuration file for shared hyperparameters across learning scripts.
"""

# --- Learning Parameters ---
num_episodes = 10000       # Number of training episodes
learning_rate = 0.1        # Learning rate (alpha)
discount_factor = 0.99     # Discount factor (gamma)

# --- Epsilon-Greedy Exploration Parameters ---
epsilon_decay_rate = 0.001 # Rate of exponential decay for epsilon
max_epsilon = 1.0          # Starting exploration rate
min_epsilon = 0.0005         # Minimum exploration rate 
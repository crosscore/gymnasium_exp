import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import os
import random # To randomly choose which Q-table to update

# env is frozenlake (deterministic version)
env = gym.make("FrozenLake-v1", render_mode="rgb_array", is_slippery=False)

# Double Q-learning parameters (using previously tuned values)
num_episodes = 60000
learning_rate = 0.1
discount_factor = 0.99
epsilon = 1.0
epsilon_decay_rate = 0.001 # Increase decay rate for faster convergence
max_epsilon = 1.0
min_epsilon = 0.0005

# Q-table initialization (Two Q-tables)
q_table_a = np.zeros((env.observation_space.n, env.action_space.n))
q_table_b = np.zeros((env.observation_space.n, env.action_space.n))

# For plotting/logging
rewards_per_episode = np.zeros(num_episodes)
average_rewards_log = []

# Create output directory if it doesn't exist
output_dir = "output/double_q_deterministic" # Changed directory name
os.makedirs(output_dir, exist_ok=True)

# --- Plotting Setup for Learning Curve ---
plt.ion()
fig_lc, ax_lc = plt.subplots(figsize=(10, 5))
line_lc, = ax_lc.plot([], [], 'm-') # Changed color to magenta
ax_lc.set_title('Double Q-Learning Progress: Average Reward per 100 Episodes') # Changed title
ax_lc.set_xlabel('Episode')
ax_lc.set_ylabel('Average Reward (Last 100 episodes)')
ax_lc.grid(True)
plt.show(block=False)

# --- Double Q-Learning Training Loop ---
for episode in range(num_episodes):
    state, info = env.reset()
    terminated = False
    truncated = False
    episode_reward = 0

    # Decide if we render this episode
    render_this_episode = (episode == 0 or (episode + 1) % 5000 == 0)
    if render_this_episode:
        if 'fig_anim' not in locals() or not plt.fignum_exists(fig_anim.number):
            fig_anim, ax_anim = plt.subplots()
        else:
            ax_anim.clear()
        frame = env.render()
        img = ax_anim.imshow(frame)
        ax_anim.set_title(f"Double Q-Learning Episode: {episode + 1}") # Changed title
        fig_anim.canvas.draw_idle()
        plt.pause(0.01)

    while not terminated and not truncated:
        if render_this_episode:
            frame = env.render()
            img.set_data(frame)
            ax_anim.set_title(f"Double Q-Learning Episode: {episode + 1}")
            fig_anim.canvas.draw_idle()
            fig_anim.canvas.flush_events()
            plt.pause(0.01)

        # Action selection using epsilon-greedy (based on the sum of both Q-tables)
        if np.random.random() < epsilon:
            action = env.action_space.sample() # Explore
        else:
            # Use the sum of Q-tables for more robust action selection
            q_sum = q_table_a[state, :] + q_table_b[state, :]
            action = np.argmax(q_sum) # Exploit

        # Step the environment
        next_state, reward, terminated, truncated, info = env.step(action)

        # Double Q-Learning Update
        if random.uniform(0, 1) < 0.5:
            # Update Q_A using Q_B's estimate
            best_next_action_a = np.argmax(q_table_a[next_state, :])
            td_target = reward + discount_factor * q_table_b[next_state, best_next_action_a] * (1 - terminated)
            td_error = td_target - q_table_a[state, action]
            q_table_a[state, action] = q_table_a[state, action] + learning_rate * td_error
        else:
            # Update Q_B using Q_A's estimate
            best_next_action_b = np.argmax(q_table_b[next_state, :])
            td_target = reward + discount_factor * q_table_a[next_state, best_next_action_b] * (1 - terminated)
            td_error = td_target - q_table_b[state, action]
            q_table_b[state, action] = q_table_b[state, action] + learning_rate * td_error

        episode_reward += reward
        state = next_state

    # Epsilon decay (Exponential decay)
    epsilon = min_epsilon + (max_epsilon - min_epsilon) * np.exp(-epsilon_decay_rate * episode)

    rewards_per_episode[episode] = episode_reward

    # --- Logging and Plotting Progress ---
    if (episode + 1) % 100 == 0:
        last_100_rewards = rewards_per_episode[episode - 99 : episode + 1]
        avg_reward = np.mean(last_100_rewards)
        average_rewards_log.append(avg_reward)
        print(f"Episode {episode + 1}/{num_episodes} | Last 100 avg reward: {avg_reward:.4f} | Epsilon: {epsilon:.4f}")

        # Update the learning curve plot
        episodes_logged = np.arange(1, len(average_rewards_log) + 1) * 100
        line_lc.set_xdata(episodes_logged)
        line_lc.set_ydata(average_rewards_log)
        ax_lc.relim()
        ax_lc.autoscale_view(True, True, True)
        fig_lc.canvas.draw_idle()
        fig_lc.canvas.flush_events()

        if (episode + 1) % 1000 == 0:
            plot_filename = os.path.join(output_dir, f"double_q_deterministic_curve_ep{episode + 1}.png") # Changed filename
            try:
                fig_lc.savefig(plot_filename)
                print(f"Saved learning curve plot to {plot_filename}")
            except Exception as e:
                print(f"Error saving plot: {e}")

    if render_this_episode and 'fig_anim' in locals() and plt.fignum_exists(fig_anim.number):
        plt.close(fig_anim)

# --- End of Training ---
env.close()
plt.ioff()

print("Double Q-Learning (Deterministic) Training finished.") # Changed message

# Save the final learning curve plot
final_plot_filename = os.path.join(output_dir, "double_q_deterministic_curve_final.png") # Changed filename
try:
    fig_lc.savefig(final_plot_filename)
    print(f"Saved final Double Q-Learning (Deterministic) curve plot to {final_plot_filename}") # Changed message
except Exception as e:
    print(f"Error saving final plot: {e}")

# Keep the learning curve plot window open
print("Close the plot window to exit.")
# plt.show(block=True) # Commented out for batch execution

plt.close() 
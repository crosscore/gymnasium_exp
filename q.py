import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import os # outputフォルダ作成のため追加

# env is frozenlake (deterministic version)
env = gym.make("FrozenLake-v1", render_mode="rgb_array", is_slippery=False)
# state, info = env.reset()
# print(f"Initial state: {state}")
# print(f"Initial info: {info}")

# Q-Learning parameters
num_episodes = 60000 # ユーザーが変更
learning_rate = 0.1
discount_factor = 0.99
epsilon = 1.0
epsilon_decay_rate = 0.001 # Increase decay rate
max_epsilon = 1.0
min_epsilon = 0.0005

# Q-table initialization
q_table = np.zeros((env.observation_space.n, env.action_space.n))

# For plotting/logging
rewards_per_episode = np.zeros(num_episodes)
average_rewards_log = [] # 平均報酬のログ

# Create output directory if it doesn't exist
output_dir = "output/q_deterministic" # Changed directory name
os.makedirs(output_dir, exist_ok=True)

# --- Plotting Setup for Learning Curve ---
plt.ion() # Turn on interactive mode for live plotting
fig_lc, ax_lc = plt.subplots(figsize=(10, 5)) # Learning Curve plot
line_lc, = ax_lc.plot([], [], 'r-') # Initialize empty plot
ax_lc.set_title('Learning Progress: Average Reward per 100 Episodes')
ax_lc.set_xlabel('Episode')
ax_lc.set_ylabel('Average Reward (Last 100 episodes)')
ax_lc.grid(True)
plt.show(block=False)

# --- Q-Learning Training Loop ---
for episode in range(num_episodes):
    state, info = env.reset()
    terminated = False
    truncated = False
    episode_reward = 0

    # Decide if we render this episode (e.g., first episode and every 5000 after)
    render_this_episode = (episode == 0 or (episode + 1) % 5000 == 0)
    if render_this_episode:
        # Setup animation plot if needed
        if 'fig_anim' not in locals() or not plt.fignum_exists(fig_anim.number):
            fig_anim, ax_anim = plt.subplots()
        else:
            ax_anim.clear() # Clear previous frame
        frame = env.render()
        img = ax_anim.imshow(frame)
        ax_anim.set_title(f"Episode: {episode + 1}")
        fig_anim.canvas.draw_idle()
        plt.pause(0.01) # Allow plot window to appear

    while not terminated and not truncated:
        # Render animation if enabled for this episode
        if render_this_episode:
            frame = env.render()
            img.set_data(frame)
            ax_anim.set_title(f"Episode: {episode + 1}")
            fig_anim.canvas.draw_idle()
            fig_anim.canvas.flush_events()
            plt.pause(0.01) # Short pause for visibility

        # Epsilon-greedy action selection
        if np.random.random() < epsilon:
            action = env.action_space.sample() # Explore
        else:
            action = np.argmax(q_table[state, :]) # Exploit

        # Step the environment
        next_state, reward, terminated, truncated, info = env.step(action)

        # Update Q-table
        best_next_action_q = np.max(q_table[next_state, :])
        td_target = reward + discount_factor * best_next_action_q * (1 - terminated)
        td_error = td_target - q_table[state, action]
        q_table[state, action] = q_table[state, action] + learning_rate * td_error

        episode_reward += reward
        state = next_state

    # Epsilon decay (change to exponential decay)
    # epsilon = max(min_epsilon, epsilon - epsilon_decay_rate) # Linear decay (original)
    epsilon = min_epsilon + (max_epsilon - min_epsilon) * np.exp(-epsilon_decay_rate * episode)

    rewards_per_episode[episode] = episode_reward

    # --- Logging and Plotting Progress ---
    # Calculate and log average reward every 100 episodes
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

        # Save the plot periodically (e.g., every 1000 episodes)
        if (episode + 1) % 1000 == 0:
            plot_filename = os.path.join(output_dir, f"learning_curve_deterministic_ep{episode + 1}.png") # Changed filename
            try:
                fig_lc.savefig(plot_filename)
                print(f"Saved learning curve plot to {plot_filename}")
            except Exception as e:
                print(f"Error saving plot: {e}")

    # Close animation window if it was used for this episode
    if render_this_episode and 'fig_anim' in locals() and plt.fignum_exists(fig_anim.number):
        plt.close(fig_anim)

# --- End of Training ---
env.close()
plt.ioff() # Turn off interactive mode

print("Q-Learning (Deterministic) Training finished.") # Changed message

# Save the final learning curve plot
final_plot_filename = os.path.join(output_dir, "learning_curve_deterministic_final.png") # Changed filename
try:
    fig_lc.savefig(final_plot_filename)
    print(f"Saved final learning curve plot to {final_plot_filename}") # Changed message
except Exception as e:
    print(f"Error saving final plot: {e}")

# Keep the learning curve plot window open until manually closed
print("Close the plot window to exit.")
plt.show(block=True)

# (Removed previous final plot code as it's now handled by the live plot)
plt.close()

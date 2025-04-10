import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import os

# env is frozenlake (deterministic version)
env = gym.make("FrozenLake-v1", render_mode="rgb_array", is_slippery=False)

# SARSA parameters (using previously tuned values, adapted for deterministic)
# For deterministic, learning can be faster, let's reduce episodes but keep other fine-tuned params
num_episodes = 60000 # Set episodes to 60000
learning_rate = 0.05
discount_factor = 0.99
epsilon = 1.0
epsilon_decay_rate = 0.0001 # 指数減衰率として使用
max_epsilon = 1.0
min_epsilon = 0.0005

# Q-table initialization (still using a Q-table)
q_table = np.zeros((env.observation_space.n, env.action_space.n))

# For plotting/logging
rewards_per_episode = np.zeros(num_episodes)
average_rewards_log = [] # 平均報酬のログ

# Create output directory if it doesn't exist
output_dir = "output/sarsa_deterministic" # Changed directory name
os.makedirs(output_dir, exist_ok=True)

# --- Plotting Setup for Learning Curve ---
plt.ion()
fig_lc, ax_lc = plt.subplots(figsize=(10, 5))
line_lc, = ax_lc.plot([], [], 'b-') # Changed color to blue for distinction
ax_lc.set_title('SARSA Learning Progress: Average Reward per 100 Episodes') # Changed title
ax_lc.set_xlabel('Episode')
ax_lc.set_ylabel('Average Reward (Last 100 episodes)')
ax_lc.grid(True)
plt.show(block=False)

# --- SARSA Training Loop ---
for episode in range(num_episodes):
    state, info = env.reset()
    terminated = False
    truncated = False
    episode_reward = 0

    # Initial action selection using epsilon-greedy
    if np.random.random() < epsilon:
        action = env.action_space.sample()
    else:
        action = np.argmax(q_table[state, :])

    # Decide if we render this episode
    render_this_episode = (episode == 0 or (episode + 1) % 5000 == 0)
    if render_this_episode:
        if 'fig_anim' not in locals() or not plt.fignum_exists(fig_anim.number):
            fig_anim, ax_anim = plt.subplots()
        else:
            ax_anim.clear()
        frame = env.render()
        img = ax_anim.imshow(frame)
        ax_anim.set_title(f"SARSA Episode: {episode + 1}") # Changed title
        fig_anim.canvas.draw_idle()
        plt.pause(0.01)

    while not terminated and not truncated:
        if render_this_episode:
            frame = env.render()
            img.set_data(frame)
            ax_anim.set_title(f"SARSA Episode: {episode + 1}")
            fig_anim.canvas.draw_idle()
            fig_anim.canvas.flush_events()
            plt.pause(0.01)

        # Step the environment with the chosen action
        next_state, reward, terminated, truncated, info = env.step(action)

        # Choose next action (A') using epsilon-greedy for the next state (S')
        if np.random.random() < epsilon:
            next_action = env.action_space.sample() # Explore
        else:
            next_action = np.argmax(q_table[next_state, :]) # Exploit

        # SARSA Update: Q(S, A) <- Q(S, A) + alpha * [R + gamma * Q(S', A') - Q(S, A)]
        # Need Q(next_state, next_action) for the update
        td_target = reward + discount_factor * q_table[next_state, next_action] * (1 - terminated)
        td_error = td_target - q_table[state, action]
        q_table[state, action] = q_table[state, action] + learning_rate * td_error

        episode_reward += reward
        state = next_state
        action = next_action # Crucial step for SARSA: update the action for the next iteration

    # Epsilon decay (Exponential decay)
    # epsilon = max(min_epsilon, epsilon - epsilon_decay_rate) # Linear decay (original)
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
            plot_filename = os.path.join(output_dir, f"sarsa_deterministic_curve_ep{episode + 1}.png") # Changed filename
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

print("SARSA (Deterministic) Training finished.") # Changed message

# Save the final learning curve plot
final_plot_filename = os.path.join(output_dir, "sarsa_deterministic_curve_final.png") # Changed filename
try:
    fig_lc.savefig(final_plot_filename)
    print(f"Saved final SARSA (Deterministic) learning curve plot to {final_plot_filename}") # Changed message
except Exception as e:
    print(f"Error saving final plot: {e}")

# Keep the learning curve plot window open
print("Close the plot window to exit.")
plt.show(block=True)

plt.close()
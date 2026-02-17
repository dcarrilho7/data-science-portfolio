from agent import QLearningAgent
from environment import GridWorld2D
import argparse
import csv
import matplotlib
from matplotlib import colors
import numpy as np
import os
import random
import sys


def configure_matplotlib_backend():
    backend = None
    if "--backend" in sys.argv:
        try:
            idx = sys.argv.index("--backend")
            backend = sys.argv[idx + 1]
        except Exception:
            backend = None

    if backend:
        try:
            matplotlib.use(backend)
            return
        except Exception:
            pass

    try:
        if sys.platform == "darwin":
            matplotlib.use("MacOSX")
    except Exception:
        pass


configure_matplotlib_backend()
import matplotlib.pyplot as plt


ACTIONS = ["UP", "DOWN", "LEFT", "RIGHT"]


def set_seed(seed):
    if seed is None:
        return
    random.seed(seed)
    np.random.seed(seed)


def run_greedy_episode(env, agent, max_steps=50):
    state = env.reset()
    total_reward = 0.0

    for _ in range(max_steps):
        # greedy action (no exploration)
        action = max(agent.Q[state], key=agent.Q[state].get)
        next_state, reward, done = env.step(action)

        total_reward += reward
        state = next_state

        if done:
            return total_reward, reward  # final reward tells goal(+10) or trap(-10)
    return total_reward, 0  # timed out


def action_symbol(a):
    return {"UP":"^", "DOWN":"v", "LEFT":"<", "RIGHT":">"}[a]


def id_to_pos(state_id, cols=5):
    return (state_id // cols, state_id % cols)


def greedy_trajectory(env, agent, max_steps=50):
    old_eps = agent.epsilon
    agent.epsilon = 0.0  # greedy

    state = env.reset()
    path = [id_to_pos(state)]
    actions_taken = []

    for _ in range(max_steps):
        action = max(agent.Q[state], key=agent.Q[state].get)
        next_state, reward, done = env.step(action)

        actions_taken.append(action)
        state = next_state
        path.append(id_to_pos(state))

        if done:
            agent.epsilon = old_eps
            return path, actions_taken, reward  # final reward indicates goal/trap

    agent.epsilon = old_eps
    return path, actions_taken, 0  # timed out


def print_grid_with_path(env, path):
    grid = [row[:] for row in env.grid]  # copy

    # mark path cells
    for (r, c) in path:
        if grid[r][c] == ".":
            grid[r][c] = "*"

    # keep start/goal/traps visible
    sr, sc = env.start
    grid[sr][sc] = "S"
    for r in range(env.rows):
        for c in range(env.cols):
            if env.grid[r][c] == "G":
                grid[r][c] = "G"
            if env.grid[r][c] == "X":
                grid[r][c] = "X"

    print("\nGrid with greedy path (*):")
    for row in grid:
        print(" ".join(row))


def moving_average(x, window=50):
    if len(x) < window:
        return x[:]  # not enough points yet
    out = []
    running_sum = sum(x[:window])
    out.append(running_sum / window)
    for i in range(window, len(x)):
        running_sum += x[i] - x[i - window]
        out.append(running_sum / window)
    return out


def build_grid_image(env, agent_pos):
    grid = np.zeros((env.rows, env.cols), dtype=int)

    for r in range(env.rows):
        for c in range(env.cols):
            cell = env.grid[r][c]
            if cell == "X":
                grid[r][c] = 1
            elif cell == "G":
                grid[r][c] = 2
            elif cell == "S":
                grid[r][c] = 3

    ar, ac = agent_pos
    grid[ar][ac] = 4
    return grid


def maybe_init_animation(env):
    cmap = colors.ListedColormap(["white", "black", "lightblue", "lightgreen", "gold"])
    norm = colors.BoundaryNorm([0, 1, 2, 3, 4, 5], cmap.N)

    fig, ax = plt.subplots()
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title("Training (live agent position)")

    image = build_grid_image(env, env.start)
    im = ax.imshow(image, cmap=cmap, norm=norm)

    plt.show(block=False)
    plt.tight_layout()
    return fig, ax, im


def train_agent(
    env,
    agent,
    episodes=3000,
    max_steps=100,
    animate=False,
    animate_episodes=3,
    animate_delay=0.05,
    epsilon_min=0.05,
    epsilon_decay=0.999
):
    rewards_per_episode = []

    if animate:
        plt.ion()
        fig, ax, im = maybe_init_animation(env)

    for episode in range(episodes):
        state = env.reset()
        done = False
        total_reward = 0.0
        steps = 0

        while not done and steps < max_steps:
            action = agent.choose_action(state)
            next_state, reward, done = env.step(action)

            agent.learn(state, action, reward, next_state, done=done)

            state = next_state
            total_reward += reward
            steps += 1

            if animate and episode < animate_episodes:
                r, c = id_to_pos(state, cols=env.cols)
                im.set_data(build_grid_image(env, (r, c)))
                plt.pause(animate_delay)

        rewards_per_episode.append(total_reward)
        agent.epsilon = max(epsilon_min, agent.epsilon * epsilon_decay)

    if animate:
        plt.ioff()

    return rewards_per_episode


def evaluate_agent(env, agent, trials=200):
    old_eps = agent.epsilon
    agent.epsilon = 0.0  # greedy for evaluation

    goals = 0
    traps = 0
    timeouts = 0
    rewards = []

    for _ in range(trials):
        total_r, final_r = run_greedy_episode(env, agent)
        rewards.append(total_r)
        if final_r == 10:
            goals += 1
        elif final_r == -10:
            traps += 1
        else:
            timeouts += 1

    agent.epsilon = old_eps
    return goals, traps, timeouts, sum(rewards) / len(rewards)


def print_policy(agent, rows=5, cols=5):
    print("\nLearned policy (best action per state):")
    for r in range(rows):
        row_syms = []
        for c in range(cols):
            s = r * cols + c
            best_a = max(agent.Q[s], key=agent.Q[s].get)
            row_syms.append(action_symbol(best_a))
        print(" ".join(row_syms))


def plot_learning_curve(rewards_per_episode, window=50):
    plt.figure()
    plt.plot(rewards_per_episode, label="Reward per episode")
    ma = moving_average(rewards_per_episode, window=window)
    plt.plot(range(len(ma)), ma, label=f"Moving average ({window})")
    plt.title("Learning Curve: Reward per Episode")
    plt.xlabel("Episode")
    plt.ylabel("Total reward")
    plt.legend()
    plt.show()


def save_rewards_csv(rewards_per_episode, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, "rewards_per_episode.csv")
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["episode", "reward"])
        for episode, reward in enumerate(rewards_per_episode, start=1):
            writer.writerow([episode, reward])
    return out_path


def save_learning_curve_plot(rewards_per_episode, output_dir, window=50):
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, "learning_curve.png")
    plt.figure()
    plt.plot(rewards_per_episode, label="Reward per episode")
    ma = moving_average(rewards_per_episode, window=window)
    plt.plot(range(len(ma)), ma, label=f"Moving average ({window})")
    plt.title("Learning Curve: Reward per Episode")
    plt.xlabel("Episode")
    plt.ylabel("Total reward")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    return out_path


def parse_args():
    parser = argparse.ArgumentParser(description="Train a Q-learning agent in GridWorld.")
    parser.add_argument("--episodes", type=int, default=3000)
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--gamma", type=float, default=0.95)
    parser.add_argument("--epsilon", type=float, default=0.3)
    parser.add_argument("--epsilon-min", type=float, default=0.05)
    parser.add_argument("--epsilon-decay", type=float, default=0.999)
    parser.add_argument("--trials", type=int, default=200)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--no-plot", action="store_true")
    parser.add_argument("--no-save-artifacts", action="store_true")
    parser.add_argument("--output-dir", type=str, default="artifacts")
    parser.add_argument("--animate", action="store_true")
    parser.add_argument("--animate-episodes", type=int, default=3)
    parser.add_argument("--animate-delay", type=float, default=0.05)
    parser.add_argument("--animate-hold", action="store_true")
    parser.add_argument("--backend", type=str, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

    env = GridWorld2D()
    state_count = env.rows * env.cols
    agent = QLearningAgent(
        states=range(state_count),
        actions=ACTIONS,
        alpha=args.alpha,
        gamma=args.gamma,
        epsilon=args.epsilon
    )

    rewards_per_episode = train_agent(
        env,
        agent,
        episodes=args.episodes,
        animate=args.animate,
        animate_episodes=args.animate_episodes,
        animate_delay=args.animate_delay,
        epsilon_min=args.epsilon_min,
        epsilon_decay=args.epsilon_decay
    )
    print("\nTraining complete.")
    print("Final epsilon:", round(agent.epsilon, 6))

    goals, traps, timeouts, avg_reward = evaluate_agent(env, agent, trials=args.trials)
    print("\nEvaluation (greedy policy):")
    print("Goal reached:", goals, "/", args.trials)
    print("Fell into trap:", traps, "/", args.trials)
    print("Timed out:", timeouts, "/", args.trials)
    print("Average total reward:", avg_reward)

    print_policy(agent, rows=env.rows, cols=env.cols)

    path, actions_taken, final_reward = greedy_trajectory(env, agent)
    print("\nGreedy best trajectory:")
    print("Path:", path)
    print("Actions:", actions_taken)
    print("Final reward:", final_reward)
    print_grid_with_path(env, path)

    if not args.no_save_artifacts:
        csv_path = save_rewards_csv(rewards_per_episode, args.output_dir)
        png_path = save_learning_curve_plot(rewards_per_episode, args.output_dir)
        print("\nSaved artifacts:")
        print("CSV:", csv_path)
        print("Plot:", png_path)

    if not args.no_plot and not args.animate:
        plot_learning_curve(rewards_per_episode)

    if args.animate and args.animate_hold:
        plt.show()


if __name__ == "__main__":
    main()

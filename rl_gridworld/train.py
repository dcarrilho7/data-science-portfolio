from agent import QLearningAgent
from environment import GridWorld2D
import argparse
import matplotlib.pyplot as plt
import random


ACTIONS = ["UP", "DOWN", "LEFT", "RIGHT"]
STATE_COUNT = 25


def set_seed(seed):
    if seed is None:
        return
    random.seed(seed)


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


def train_agent(env, agent, episodes=3000, max_steps=100):
    rewards_per_episode = []

    for _ in range(episodes):
        state = env.reset()
        done = False
        total_reward = 0.0
        steps = 0

        while not done and steps < max_steps:
            action = agent.choose_action(state)
            next_state, reward, done = env.step(action)

            agent.learn(state, action, reward, next_state)

            state = next_state
            total_reward += reward
            steps += 1

        rewards_per_episode.append(total_reward)

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


def parse_args():
    parser = argparse.ArgumentParser(description="Train a Q-learning agent in GridWorld.")
    parser.add_argument("--episodes", type=int, default=3000)
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--gamma", type=float, default=0.95)
    parser.add_argument("--epsilon", type=float, default=0.3)
    parser.add_argument("--trials", type=int, default=200)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--no-plot", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

    env = GridWorld2D()
    agent = QLearningAgent(
        states=range(STATE_COUNT),
        actions=ACTIONS,
        alpha=args.alpha,
        gamma=args.gamma,
        epsilon=args.epsilon
    )

    rewards_per_episode = train_agent(env, agent, episodes=args.episodes)
    print("\nTraining complete.")

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

    if not args.no_plot:
        plot_learning_curve(rewards_per_episode)


if __name__ == "__main__":
    main()

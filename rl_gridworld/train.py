# train.py
from agent import QLearningAgent
from environment import GridWorld2D
import matplotlib.pyplot as plt


states = range(25)
actions = ["UP", "DOWN", "LEFT", "RIGHT"]

agent = QLearningAgent(
    states=states,
    actions=actions,
    alpha=0.1,
    gamma=0.95,
    epsilon=0.3
)

env = GridWorld2D()
episodes = 3000

for episode in range(episodes):
    state = env.reset()
    done = False

    while not done:
        action = agent.choose_action(state)
        next_state, reward, done = env.step(action)
        agent.learn(state, action, reward, next_state)
        state = next_state

print("\nTraining complete.")

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

# Evaluate
old_eps = agent.epsilon
agent.epsilon = 0.0  # turn off exploration for evaluation

trials = 200
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

agent.epsilon = old_eps  # restore

print("\nEvaluation (greedy policy):")
print("Goal reached:", goals, "/", trials)
print("Fell into trap:", traps, "/", trials)
print("Timed out:", timeouts, "/", trials)
print("Average total reward:", sum(rewards) / len(rewards))


def action_symbol(a):
    return {"UP":"^", "DOWN":"v", "LEFT":"<", "RIGHT":">"}[a]

print("\nLearned policy (best action per state):")
for r in range(5):
    row_syms = []
    for c in range(5):
        s = r * 5 + c
        best_a = max(agent.Q[s], key=agent.Q[s].get)
        row_syms.append(action_symbol(best_a))
    print(" ".join(row_syms))


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

path, actions_taken, final_reward = greedy_trajectory(env, agent)

print("\nGreedy best trajectory:")
print("Path:", path)
print("Actions:", actions_taken)
print("Final reward:", final_reward)

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

print_grid_with_path(env, path)


rewards_per_episode = []

for episode in range(episodes):
    state = env.reset()
    done = False
    total_reward = 0.0
    steps = 0
    max_steps = 100  # prevents endless wandering

    while not done and steps < max_steps:
        action = agent.choose_action(state)
        next_state, reward, done = env.step(action)

        agent.learn(state, action, reward, next_state)

        state = next_state
        total_reward += reward
        steps += 1

    rewards_per_episode.append(total_reward)


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

plt.figure()
plt.plot(rewards_per_episode, label="Reward per episode")

ma_window = 50
ma = moving_average(rewards_per_episode, window=ma_window)
plt.plot(range(len(ma)), ma, label=f"Moving average ({ma_window})")

plt.title("Learning Curve: Reward per Episode")
plt.xlabel("Episode")
plt.ylabel("Total reward")
plt.legend()
plt.show()


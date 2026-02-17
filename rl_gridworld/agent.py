# agent.py
import random

class QLearningAgent:
    def __init__(self, states, actions, alpha=0.9, gamma=0.9, epsilon=0):
        self.actions = actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon

        # Initialize Q-table
        self.Q = {
            s: {a: 0.0 for a in actions}
            for s in states
        }

    def choose_action(self, state):
        # ε-greedy policy
        if random.random() < self.epsilon:
            return random.choice(self.actions)
        return max(self.Q[state], key=self.Q[state].get)

    def learn(self, state, action, reward, next_state, done=False):
        if done:
            target = reward
        else:
            best_next_q = max(self.Q[next_state].values())
            target = reward + self.gamma * best_next_q

        self.Q[state][action] += self.alpha * (target - self.Q[state][action])

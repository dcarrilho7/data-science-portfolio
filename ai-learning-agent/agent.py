import torch
import torch.nn as nn
import torch.optim as optim
import random


class Agent:
    def __init__(self, lr=1e-3, gamma=0.99):
        pass

    def select_action(self, state, epsilon):
        pass

    def train_step(self, state, action, reward, next_state, done):
        pass

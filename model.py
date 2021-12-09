import torch

import torch.nn as nn
import torch.nn.functional as func
import torch.optim as optim

from torch.distributions import Categorical


class OneHiddenNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.fcn = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(input_size, hidden_size),
            nn.LeakyReLU(),

            nn.Dropout(0.1),
            nn.Linear(hidden_size, output_size)
        )

    def forward(self, state):
        return func.softmax(self.fcn(state), dim=0)


class TwoHiddenNetwork(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, output_size):
        super().__init__()
        self.fcn = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(input_size, hidden_size1),
            nn.LeakyReLU(),

            nn.Dropout(0.1),
            nn.Linear(hidden_size1, hidden_size2),

            nn.Dropout(0.1),
            nn.Linear(hidden_size2, output_size)
        )

    def forward(self, state):
        return func.softmax(self.fcn(state), dim=0)


class PolicyGradientAgent:
    def __init__(self, network, num_actions):
        self.network = network
        self.num_actions = num_actions
        self.optimizer = optim.AdamW(self.network.parameters(), lr=4e-3)

    def forward(self, state):
        return self.network(state)

    def learn(self, log_probs, rewards):
        loss = (-log_probs * rewards).sum()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def sample(self, state):
        action_prob = self.network(state)
        distribution = Categorical(action_prob)
        actions = set()
        while len(actions) < self.num_actions:
            actions.add(distribution.sample())
        log_probs = [distribution.log_prob(action) for action in actions]
        actions = [action.item() for action in actions]
        return actions, log_probs

    def save(self, path):
        agent_dict = {
            "network": self.network.state_dict(),
            "optimizer": self.optimizer.state_dict()
        }
        torch.save(agent_dict, path)

    def load(self, path):
        checkpoint = torch.load(path)
        self.network.load_state_dict(checkpoint['network'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])

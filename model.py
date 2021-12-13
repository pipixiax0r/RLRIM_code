import torch
import torch.nn as nn
import torch.nn.functional as func
import torch.optim as optim
from torch_geometric.nn import GATConv
from torch.distributions import Categorical


class GATNetwork(nn.Module):
    def __init__(self, conv_size, input_size, hidden_size, output_size):
        super().__init__()
        self.conv1 = GATConv(1, conv_size)
        self.conv2 = GATConv(conv_size, 1)
        self.fcn = nn.Sequential(
            nn.Dropout(0.15),
            nn.Linear(input_size, hidden_size),
            nn.LeakyReLU(),

            nn.Linear(hidden_size, output_size)
        )

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.conv1(x, edge_index)
        x = func.leaky_relu(x)
        x = self.conv2(x, edge_index)
        x = self.fcn(x.T.squeeze())
        return func.softmax(x, dim=0)


class PolicyGradientAgent:
    def __init__(self, network, num_actions):
        self.network = network
        self.num_actions = num_actions
        self.optimizer = optim.AdamW(self.network.parameters(), lr=5e-3)

    def learn(self, log_probs, rewards):
        loss = (-log_probs * rewards).sum()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def sample(self, data):
        action_prob = self.network(data)
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

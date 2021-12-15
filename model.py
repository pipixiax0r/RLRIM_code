import torch
import torch.nn as nn
import torch.nn.functional as func
import torch.optim as optim
from torch_geometric.nn import GCNConv, GATv2Conv


class GATNetwork(nn.Module):
    def __init__(self, conv_size, input_size, hidden_size, output_size):
        super().__init__()
        self.conv1 = GCNConv(1, conv_size)
        self.conv2 = GCNConv(conv_size, 1)
        self.fc = nn.Linear(input_size, output_size)
        self.fcn = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.LeakyReLU(),

            nn.Linear(hidden_size, output_size)
        )

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.conv1(x, edge_index)
        x = func.leaky_relu(x)

        x = self.conv2(x, edge_index)
        x = func.leaky_relu(x)

        x = self.fcn(x.T.squeeze())
        return func.log_softmax(x, dim=0)


class PolicyGradientAgent:
    def __init__(self, network, num_actions):
        self.network = network
        self.num_actions = num_actions
        self.optimizer = optim.AdamW(self.network.parameters(), lr=1e-3)

    def learn(self, log_probs, rewards):
        loss = (-log_probs * rewards).sum()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def sample(self, data):
        action_probs = torch.exp(self.network(data))
        idx = action_probs.multinomial(num_samples=self.num_actions)
        actions = torch.arange(len(action_probs))[idx]
        log_probs = torch.log(action_probs[idx])
        return actions.detach().cpu().numpy().tolist(), log_probs

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

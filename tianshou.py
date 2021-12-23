import gym
import numpy as np
import diffusion_gym
import networkx as nx
import torch
import torch.nn as nn
import


class Net(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.model = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(input_size, input_size*2),
            nn.ReLU(inplace=True),

            nn.Dropout(0.1),
            nn.Linear(input_size*2, input_size*2),
            nn.ReLU(inplace=True),

            nn.Dropout(0.1),
            nn.Linear(input_size*2, input_size*2),
            nn.ReLU(inplace=True),

            nn.Dropout(0.1),
            nn.Linear(input_size*2, output_size)
        )

    def forward(self, obs, state=None, info={}):
        if not isinstance(obs, torch.Tensor):
            obs = torch.tensor(obs, dtype=torch.float)
        batch = obs.shape[0]
        logits = self.model(obs.view(batch, -1))
        return logits, state


graph = nx.karate_club_graph()
seeds = np.array([0, 5, 7, 11, 13, 22])
blockers = np.arange(graph.number_of_nodes())
env = gym.make('ic_env-v0', graph=graph, seeds=seeds, blockers=blockers)

net = Net(state_shape, action_shape)
optim = torch.optim.Adam(net.parameters(), lr=1e-3)
policy = ts.policy.DQNPolicy(net, optim, discount_factor=0.9, estimation_step=3, target_update_freq=320)

import pickle

import torch
import numpy as np
import pandas as pd
import networkx as nx
from tqdm import tqdm
from functools import reduce
from env import Env
from model import PolicyGradientAgent, EmailNetwork
from torch_geometric.data import Data
from torch_geometric.datasets import KarateClub


def idx2array(idx: int, length: int) -> np.ndarray:
    array = np.zeros(length).astype(np.bool_)
    array[idx] = 1
    return array


with open('nxGraph/graph_email', 'rb') as f:
    graph = pickle.load(f)


num_nodes = len(graph.nodes)

seed_candidate = [481, 171, 51, 182, 103]
num_blocker = 5
blocker_deg = 35
num_batch = 600
episode_per_batch = 30
num_diffusion = 5
window_size = 3
decay = 0.5
device = torch.device('cpu')

env = Env(graph, seed_candidate, blocker_deg)
print(env.model.prob_matrix)
print(env.seed_candidate)
print(f'num of blocker candidate : {len(env.blocker_candidate)}')

network = EmailNetwork(4, num_nodes, 25, len(env.blocker_candidate))
network = network.to(device)
network.device = device
agent = PolicyGradientAgent(network, num_blocker)
agent.network.train()

rewards_plot = []
infected_plot = []
batch_bar = tqdm(range(num_batch))
edge_index = torch.tensor([[x[0] for x in graph.edges], [x[1] for x in graph.edges]], dtype=torch.long, device=device)

for batch in batch_bar:
    batch_rewards, batch_probs = [], []
    avg_reward, avg_infected = 0, 0

    for episode in range(episode_per_batch):
        state = env.reset()
        episode_rewards, episode_probs = [], []

        for i in range(num_diffusion):
            x = torch.tensor([[i] for i in state], dtype=torch.float, device=device)
            actions, log_probs = agent.sample(Data(x=x, edge_index=edge_index.contiguous()))
            if batch % 100 == 0 and episode == 0:
                print(actions)
            state, reward, done = env.step(actions)
            episode_probs += log_probs
            episode_rewards.append(reward)
            if done:
                break

        n = len(episode_rewards)
        decay_rewards = [0 for i in range(n)]
        for i in range(n):
            for j in range(i, n):
                decay_rewards[i] += decay**(j-i)*episode_rewards[j]
        # 收益和对应的action概率长度一致
        decay_rewards = [[reward for _ in range(num_blocker)] for reward in decay_rewards]
        decay_rewards = list(reduce(lambda x, y: x + y, decay_rewards))
        batch_rewards = batch_rewards + decay_rewards
        batch_probs = batch_probs + episode_probs
        avg_reward += sum(episode_rewards) / episode_per_batch
        avg_infected += sum(env.model.state) / episode_per_batch

    rewards_plot.append(avg_reward)
    infected_plot.append(avg_infected)
    batch_bar.set_description(f"avg_reward: {avg_reward: 4.2f}, avg_infected: {avg_infected: 4.2f}")
    batch_probs = torch.stack(batch_probs)
    batch_rewards = torch.tensor(batch_rewards)
    agent.learn(batch_probs.to(device), batch_rewards.to(device))

df = pd.DataFrame({'rewards': rewards_plot, 'infected': infected_plot})
df.to_csv(f'karate_seeds_blockers{num_blocker}.csv', index=False)
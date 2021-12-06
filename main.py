import torch
import numpy as np
import pandas as pd
import seaborn as sns
import networkx as nx
from tqdm import tqdm
from env import Env
from model import PolicyGradientAgent, PolicyGradientNetwork


def idx2array(idx: int, length: int) -> np.ndarray:
    array = np.zeros(length).astype(np.bool_)
    array[idx] = 1
    return array


graph = nx.karate_club_graph()
num_nodes = len(graph.nodes)

num_seeds = 1
num_blocker = 1
seeds_deg = 8
blocker_deg = 5
num_batch = 1000
episode_per_batch = 40
num_diffusion = 5
window_size = 3
decay = 0.5

env = Env(graph, num_seeds, seeds_deg, blocker_deg)
print(env.model.prob_matrix)
print(env.seed_candidate)
print(f'num of blocker candidate : {len(env.blocker_candidate)}')

network = PolicyGradientNetwork(num_nodes, 20, len(env.blocker_candidate))
agent = PolicyGradientAgent(network)
agent.network.train()

rewards_plot = []
infected_plot = []
batch_bar = tqdm(range(num_batch))

for batch in batch_bar:
    batch_rewards, batch_probs = [], []
    avg_reward, avg_infected = 0, 0
    for episode in range(episode_per_batch):
        state = env.reset()
        episode_rewards, episode_probs = [], []

        for i in range(num_diffusion):
            action, log_prob = agent.sample(state)
            state, reward, done = env.step(action)
            episode_rewards.append(reward)
            episode_probs.append(log_prob)
            if done:
                break

        n = len(episode_rewards)
        decay_rewards = [0 for i in range(n)]
        for i in range(n):
            for j in range(i, n):
                decay_rewards[i] += decay**(j-i)*episode_rewards[j]

        batch_rewards = batch_rewards + decay_rewards
        batch_probs = batch_probs + episode_probs
        avg_reward += sum(episode_rewards) / episode_per_batch
        avg_infected += sum(env.model.state) / episode_per_batch

    rewards_plot.append(avg_reward)
    infected_plot.append(avg_infected)
    batch_bar.set_description(f"avg_reward: {avg_reward: 4.2f}, avg_infected: {avg_infected: 4.2f}")
    batch_rewards = np.array(batch_rewards)
    agent.learn(torch.stack(batch_probs), torch.from_numpy(batch_rewards))

df = pd.DataFrame({'rewards': rewards_plot, 'infected': infected_plot})
df.to_csv(f'karate_seeds{num_seeds}_blockers{num_blocker}_deg{blocker_deg}.csv', index=False)

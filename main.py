import numpy as np
import networkx as nx
from tqdm import tqdm
import torch

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
blocker_deg = 4
num_batch = 500
episode_per_batch = 5
num_diffusion = 5
decay = 0.5

env = Env(graph, num_seeds, seeds_deg, blocker_deg)
print(env.model.prob_matrix)
print(env.seed_candidate)
print(f'num of blocker candidate : {len(env.blocker_candidate)}')

network = PolicyGradientNetwork(num_nodes, 25, len(env.blocker_candidate))
agent = PolicyGradientAgent(network)
agent.network.train()

batch_bar = tqdm(range(num_batch))
for batch in batch_bar:
    episode_reward = 0
    batch_rewards, batch_probs = [], []

    for episode in range(episode_per_batch):
        rewards, log_probs = [], []
        state = env.reset()
        for i in range(num_diffusion):
            action, log_prob = agent.sample(state)
            state, reward, done = env.step(action)

            log_probs.append(log_prob)
            if len(rewards) >= 3:
                for j in range(len(rewards)-3, len(rewards)):
                    rewards[j] += (decay ** (len(rewards)-j)) * reward
            rewards.append(reward)

            if done:
                break

        episode_reward += int(sum(rewards))/episode_per_batch
        batch_rewards = batch_rewards + rewards
        batch_probs = batch_probs + log_probs

    batch_bar.set_description(f"Total: {int(episode_reward): 4.1f}")
    agent.learn(torch.stack(batch_probs), torch.from_numpy(batch_rewards))

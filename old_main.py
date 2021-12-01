import torch
import pickle
import numpy as np
import pandas as pd
import networkx as nx

from env import Env
from tqdm import tqdm
from model import PolicyGradientNetwork, PolicyGradientAgent


# 距离种子节点距离为n的邻居
def n_depth_neighbors(graph, source, depth=1):
    neighbors = []
    for neighbor in dict(nx.bfs_successors(graph, source, depth)).values():
        neighbors = neighbors + neighbor
    return neighbors


dataset_name = 'email'
num_seeds = 3           # 初始被激活的节点数
seed_deg = 100          # 被选为种子节点的最小节点数
blocker_deg = 100       # 被选为Blocker节点的最小节点数
num_blocker = 3         # 每回合禁用节点数
num_batch = 5000        # 更新次数
num_diffusion = 5       # 网络迭代轮数
episode_per_batch = 5   # 每次更新使用的数据量
decay = 0.7             # 奖励衰减率

with open(f'nxGraph/graph_{dataset_name}', 'rb') as f:
    graph = pickle.load(f)
    prob_matrix = nx.adjacency_matrix(graph).todense().astype(np.float16)
df_edge = pd.read_csv(f'Datasets/{dataset_name}_edge.csv')
df_node = pd.read_csv(f'Datasets/{dataset_name}_node.csv')

seeds = df_node[df_node['in_deg'] > seed_deg]['in_deg']

candidates = []
for seed in seeds:
    candidates = candidates + n_depth_neighbors(graph, seed, 1)
candidates = list(set(candidates))
df_candidates = df_node.iloc[candidates]
df_candidates = df_candidates[df_candidates['out_deg'] > blocker_deg]
candidates = list(df_candidates['id'])

network = PolicyGradientNetwork(len(df_node), len(df_node)//4, len(candidates))
agent = PolicyGradientAgent(network)
env = Env(graph, df_node, num_seeds)

agent.network.train()

avg_total_rewards = []
batch_bar = tqdm(range(num_batch))

for batch in batch_bar:
    log_probs, seq_rewards = [], []
    total_rewards = []

    for episode in range(episode_per_batch):
        state = env.reset()
        total_reward = 0

        for i in range(num_diffusion):
            action, log_prob = agent.sample(state)
            next_state, reward, done = env.step(action)

            log_probs.append(log_prob)

            # accumulative decaying reward
            for j in range(len(seq_rewards)):
                seq_rewards[j] += decay ** (len(seq_rewards) - j) * reward
            seq_rewards.append(reward)
            total_reward += reward
            state = next_state

            if done:
                break

        total_rewards.append(total_reward)

    avg_total_reward = sum(total_rewards) / len(total_rewards)
    batch_bar.set_description(f"Total: {avg_total_reward: 4.1f}")

    agent.learn(torch.stack(log_probs), torch.FloatTensor(seq_rewards))

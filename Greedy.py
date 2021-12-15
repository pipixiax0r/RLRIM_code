import numpy as np
import pickle

from env import Env, DiffusionEnd
import networkx as nx
from utils import one_hot2idx, deg

# with open('nxGraph/graph_email', 'rb') as f:
#     graph = pickle.load(f)
#     num_nodes = len(graph.nodes)
graph = nx.karate_club_graph()

num_seeds = 1
num_blocker = 1
seeds_deg = 8
blocker_deg = 0
num_episode = 1000
num_diffusion = 10

env = Env(graph, num_seeds, seeds_deg, blocker_deg)
avg_reward = 0
avg_infected = 0

for episode in range(num_episode):
    state = env.reset()
    episode_rewards = []
    for i in range(num_diffusion):
        try:
            candidates = one_hot2idx(env.valid_blocker())
            if len(candidates) == 0:
                break
            candidates_deg = [(i, deg(d, graph)) for i, d in enumerate(candidates)]
            candidates_deg.sort(key=lambda x: x[1], reverse=True)
            blocker = [x[0] for x in candidates_deg][:num_blocker]
            state, reward, done = env.step(blocker)
            episode_rewards.append(reward)

        except DiffusionEnd:
            done = 1

        if done == 1:
            break

    avg_reward += sum(episode_rewards) / num_episode
    avg_infected += sum(env.model.state) / num_episode

print(f'avg_reward:{avg_reward}\tavg_infected:{avg_infected}')
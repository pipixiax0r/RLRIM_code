import numpy as np
import pandas as pd
import networkx as nx

from diffusion import DiffusionEnd
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

env = Env(graph, num_seeds, seeds_deg, blocker_deg)
print(env.model.prob_matrix)
print(env.seed_candidate)
print(f'num of blocker candidate : {len(env.blocker_candidate)}')

network = PolicyGradientNetwork(num_nodes, 25, len(env.blocker_candidate))
agent = PolicyGradientAgent(network)

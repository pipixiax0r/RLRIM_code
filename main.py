import numpy as np
import pandas as pd
import networkx as nx

from diffusion import DiffusionEnd
from env import Env


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
env.reset(idx2array(env.seed_candidate[0], num_nodes))
print(env.seeds)
try:
    for i in range(5):
        print(env.step())
        print(env.model.state)
except DiffusionEnd:
    pass

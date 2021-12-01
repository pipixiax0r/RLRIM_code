import networkx as nx
from random import sample
from typing import Union, List

import numpy as np

from diffusion import ICModel
from utils import deg


class Env:
    def __init__(self, graph: Union[nx.Graph, nx.DiGraph], num_seeds: int, seed_min_deg: int, blocker_min_deg: int):
        """
        谣言传播的强化学习环境

        """
        self.graph = graph
        self.num_seeds = num_seeds
        self.model = ICModel(self.graph)
        self.seeds = []
        self.blocker_seq = []

        if isinstance(graph, nx.Graph):
            self.seed_candidate = list(filter(lambda x: deg(x, graph) >= seed_min_deg, self.graph.nodes()))
            self.blocker_candidate = list(filter(lambda x: deg(x, graph) >= blocker_min_deg, self.graph.nodes()))
        elif isinstance(graph, nx.DiGraph):
            self.seed_candidate = list(filter(lambda x: deg(x, graph)[1] >= seed_min_deg, self.graph.nodes()))
            self.blocker_candidate = list(filter(lambda x: deg(x, graph)[1] >= blocker_min_deg, self.graph.nodes()))
        else:
            raise TypeError(f'not supported for the input types: {type(graph)}')

    def reset(self, seeds: np.ndarray = None) -> np.array:
        self.model = ICModel(self.graph)

        if not isinstance(seeds, np.ndarray):
            seeds = sample(self.seed_candidate, self.num_seeds)
        else:
            if not seeds.any():
                seeds = sample(self.seed_candidate, self.num_seeds)

        self.seeds = seeds
        self.model.state[seeds] = 1
        self.model.active[seeds] = 1

        return self.model.state

    def step(self, blocker: Union[List, np.array] = None) -> np.array:
        if blocker is None:
            blocker = np.zeros(self.num_seeds).astype(np.bool_)
        state, active = self.model.diffusion(blocker)
        reward = -sum(active) - sum(blocker)*0.5
        return reward

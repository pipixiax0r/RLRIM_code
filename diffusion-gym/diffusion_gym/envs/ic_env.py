import gym
import numpy as np
import networkx as nx
from diffusion import DiffusionEnd, ICModel
from functools import reduce
from typing import *


class ICEnv(gym.Env):
    def __init__(self, graph: Union[nx.Graph, nx.DiGraph], seeds: np.ndarray, blockers: np.ndarray):
        if not isinstance(graph, (nx.Graph, nx.DiGraph)):
            raise TypeError(f'not supported for the input types: {type(graph)}')
        self.graph = graph
        self.seeds = seeds
        self.blockers = blockers
        self.model = ICModel(self.graph)
        self.blocker_seq = []

    def _block_time_loss(self):
        """
        预测节点连续封禁的损失
        :return:
        """
        if len(self.blocker_seq) < 2:
            return 0
        return len(reduce(lambda x, y: set(x) & set(y), self.blocker_seq[-2:]))

    def _block_invalid_loss(self, blocker_one_hot: np.ndarray):
        """
        预测节点无效时的损失
        """
        valid_blocker = self.valid_blocker()
        return np.sum(blocker_one_hot & (~valid_blocker)) - np.sum(blocker_one_hot & valid_blocker)

    def valid_blocker(self):
        try:
            valid_blocker = reduce(lambda x, y: x | y, self.model.prob_matrix[self.model.active].astype(np.bool_)) & (~self.model.state)
        except TypeError:
            raise DiffusionEnd()
        return valid_blocker

    def reset(self):
        self.model.reset()
        seed = np.random.choice(self.seeds)

        self.model.state[seed] = 1
        self.model.active[seed] = 1
        return np.concatenate(self.model.state, self.model.active), 0, 0, {}

    def step(self, action: Union[int, List]):
        if isinstance(action, (list, int)):
            blocker = self.blockers[action]
        else:
            raise TypeError(f'not supported for the input types: {type(action)}')

        self.blocker_seq.append(blocker)
        blocker_one_hot = np.zeros(self.model.num_nodes).astype(np.bool_)
        blocker_one_hot[blocker] = 1
        block_invalid_loss = self._block_invalid_loss(blocker_one_hot)
        state, active = self.model.diffusion(blocker_one_hot)
        reward = -np.sum(active) - self._block_time_loss()*2 - block_invalid_loss()*3
        done = 0 if self.model.active.any() else 1
        return np.concatenate(self.model.state, self.model.active), reward, done, {}

    def render(self, mode="human"):
        pass

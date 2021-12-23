import networkx as nx
import numpy as np
from random import sample
from typing import Union, Tuple, List
from functools import reduce

from diffusion import ICModel
from utils import deg
from diffusion import DiffusionEnd


class Env:
    def __init__(self, graph: Union[nx.Graph, nx.DiGraph], seeds: List, num_blocker: int):
        """
        谣言传播的强化学习环境
        """
        self.graph = graph
        self.model = ICModel(self.graph)
        self.seeds = seeds
        self.num_blocker = num_blocker
        self.blocker_seq = []

    def _block_time_loss(self):
        """
        预测节点连续封禁的损失
        :return:
        """
        if len(self.blocker_seq) < 2:
            return 0
        return len(reduce(lambda x, y: set(x) & set(y), self.blocker_seq[-2:]))*2

    def _block_invalid_loss(self, blocker_one_hot):
        """
        预测节点无效时的损失
        """
        valid_blocker = self.valid_blocker()
        return np.sum(blocker_one_hot & (~valid_blocker)) - np.sum(blocker_one_hot & valid_blocker)*0.5

    def valid_blocker(self):
        try:
            valid_blocker = reduce(lambda x, y: x | y, self.model.prob_matrix[self.model.active].astype(np.bool_)) & (~self.model.state)
        except TypeError:
            raise DiffusionEnd()
        return valid_blocker

    def reset(self, seed: np.ndarray = None, random_seed: int = None) -> np.array:
        self.model.reset()
        if not isinstance(seed, np.ndarray):
            np.random.seed(random_seed)
            seed = sample(self.seed_candidate, 1)
        else:
            if not seed.any():
                np.random.seed(random_seed)
                seed = sample(self.seed_candidate, 1)

        self.seed = seed
        self.model.state[seed] = 1
        self.model.active[seed] = 1

        return self.model.state

    def step(self, action: np.ndarray):
        action = action / np.sum(action)
        np.random.choice(np.arange(action.shape[0]), self.num_blocker, p=)

        # if blocker is None:
        #     blocker = np.zeros(self.graph.number_of_nodes()).astype(np.bool_)
        # elif isinstance(blocker, (list, int)):
        #     blocker = self.blocker_candidate[blocker]
        # else:
        #     raise TypeError(f'not supported for the input types: {type(blocker)}')
        #
        # try:
        #     blocker_one_hot = np.zeros(self.model.num_nodes).astype(np.bool_)
        #     blocker_one_hot[blocker] = 1
        #     self.blocker_seq.append(blocker)
        #     block_invalid_loss = self._block_invalid_loss(blocker_one_hot)  # 必须在演化之前执行
        #     state, active = self.model.diffusion(blocker_one_hot)
        #     reward = - np.sum(active) - self._block_time_loss() - block_invalid_loss
        #     done = 0
        # except DiffusionEnd:
        #     state = self.model.state
        #     reward = 0
        #     done = 1

        # return state, reward, done

import numpy as np

from typing import Tuple
from utils import graph2adj_matrix


def prob2state(prob: np.array) -> np.array:
    """
    Get the state of the node after diffusion according to the probability

    :param prob: probability of being activated
    :return: state array
    """
    return np.where(np.random.uniform(-1, 0, size=prob.shape)+prob >= 0, 1, 0).astype(np.bool_)


class DiffusionEnd(Exception):
    pass


class ICModel:
    def __init__(self, graph):
        self.graph = graph
        self.prob_matrix = graph2adj_matrix(graph)
        self.num_nodes = len(graph.nodes())

        self.state = np.zeros(self.num_nodes).astype(np.bool_)
        self.active = np.zeros(self.num_nodes).astype(np.bool_)

    def diffusion(self, blocker: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        independent cascade process
        """
        if not self.active.any():
            raise DiffusionEnd()

        temp_state = np.zeros_like(self.state).astype(np.bool_)

        if isinstance(blocker, np.ndarray):
            blocker = blocker.astype(np.bool_)
        else:
            blocker = np.zeros_like(self.state).astype(np.bool_)

        blocker = (~self.state) & blocker
        for prob in self.prob_matrix[self.active == 1, :]:
            temp_state = prob2state(prob) | temp_state

        temp_state = temp_state & (~blocker)
        self.active = temp_state & (~self.state)
        self.state = temp_state | self.state

        return self.state, self.active


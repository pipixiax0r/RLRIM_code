import numpy as np
import networkx as nx
from typing import Union, Tuple, List


def deg(node: int, graph: Union[nx.Graph, nx.DiGraph]) -> Union[int, Tuple[int, int]]:
    if isinstance(graph, nx.Graph):
        return len(graph[node])
    elif isinstance(graph, nx.DiGraph):
        in_deg = len(list(graph.predecessors(node)))
        out_deg = len(list(graph.successors(node)))
        return in_deg, out_deg
    else:
        raise TypeError(f'not supported for the input types: {type(graph)}')


def graph2adj_matrix(graph: Union[nx.Graph, nx.DiGraph]) -> np.array:
    num_nodes = len(graph.nodes())
    adj_matrix = np.zeros((num_nodes, num_nodes)).astype(np.float)

    if isinstance(graph, nx.Graph):
        for a, b in graph.edges():
            deg_a = deg(a, graph)
            deg_b = deg(b, graph)
            adj_matrix[a][b] = np.log(deg_a) / (np.log(deg_a * deg_b) + 1)
            adj_matrix[b][a] = np.log(deg_b) / (np.log(deg_a * deg_b) + 1)
    elif isinstance(graph, nx.DiGraph):
        for a, b in graph.edges():
            deg_a = deg(a, graph)
            deg_b = deg(b, graph)
            adj_matrix[a][b] = np.log(deg_a) / (np.log(deg_a * deg_b) + 1)
    else:
        raise TypeError(f'not supported for the input types: {type(graph)}')

    return adj_matrix


def one_hot2idx(array: np.ndarray) -> List:
    return list(np.arange(len(array))[array])


def n_depth_neighbors(graph, source, depth=1):
    neighbors = []
    for neighbor in dict(nx.bfs_successors(graph, source, depth)).values():
        neighbors = neighbors + neighbor
    return neighbors

from env import Env
import networkx as nx


graph = nx.karate_club_graph()
num_seeds = 1
num_blocker = 1
seeds_deg = 8
blocker_deg = 5
num_diffusion = 5

env = Env(graph, num_seeds, seeds_deg, blocker_deg)
for i in range(num_diffusion):
    state = env.reset()


import numpy as np
import igraph

from ..utils import P_Dict
from ...utils import l2_objective_function_diss_matrix


def st_decoder(chromosome: np.ndarray, diss_matrix: np.ndarray,
                num_edges: int, K: int,
                G: igraph.Graph, diss_weights: np.ndarray,) -> P_Dict:

    
    # Get edge weigths from the first part
    edge_weigths = chromosome[:num_edges] * diss_weights

    # Get K seed nodes from the second part (lowest values)
    seed_nodes = np.argsort(chromosome[num_edges:])[:K]

    # run dijkstra from each seed node
    dist_from_seeds = G.distances(source = seed_nodes,
                                  weights = edge_weigths,
                                  algorithm="dijkstra")
    
    # Assign each node to the index of the closest seed 
    P = {k: [] for k in range(1, K + 1)}
    for node in range(G.vcount()):
        # dict of distances from seeds to the node
        distances = {(idx+1): dist_from_seeds[idx][node]
                        for idx in range(K)}
        # select the closest 
        k_star = min(distances, key = lambda k: (distances[k], k))
        P[k_star].append(node)

    return P


def st_fitness(chromosome: np.ndarray, diss_matrix: np.ndarray,
                num_edges: int, K: int,
                G: igraph.Graph, diss_weights: np.ndarray,) -> float:
    P = st_decoder(chromosome, diss_matrix, num_edges, K, G, diss_weights)
    return l2_objective_function_diss_matrix(P, diss_matrix)







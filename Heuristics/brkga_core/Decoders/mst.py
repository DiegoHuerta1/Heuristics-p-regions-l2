import numpy as np
import igraph

from ..utils import P_Dict
from ...utils import l2_objective_function_diss_matrix


def mst_decoder(chromosome: np.ndarray, diss_matrix: np.ndarray,
                num_edges: int, K: int,
                G: igraph.Graph, diss_weights: np.ndarray,) -> P_Dict:

    # Get two weights from the chromosome 
    w_minus = chromosome[:num_edges] * diss_weights
    w_plus = chromosome[num_edges:]

    # Get edges from a minimum spanning tree using w_minus
    mst_edges_id = G.spanning_tree(weights = w_minus, return_tree = False)

    # Drop the first K-1 edges (considering the weights in w_plus)
    mst_edges_id.sort(key = lambda e: w_plus[e], reverse = True)
    final_edges_id = mst_edges_id[(K - 1):]

    # Remove all edges from the graph, excpet the final edges
    edges_2_remove = set(range(num_edges)) - set(final_edges_id)
    G_copy = G.copy()
    G_copy.delete_edges(edges_2_remove)

    # Make the partition using the connected components
    components = G_copy.connected_components()
    P = {idx+1: nodes for idx, nodes in enumerate(components)}
    return P


def mst_fitness(chromosome: np.ndarray, diss_matrix: np.ndarray,
                num_edges: int, K: int,
                G: igraph.Graph, diss_weights: np.ndarray,) -> float:
    P = mst_decoder(chromosome, diss_matrix, num_edges, K, G, diss_weights)
    return l2_objective_function_diss_matrix(P, diss_matrix)







import numpy as np
import igraph

from ..utils import P_Dict
from ...utils import l2_objective_function_diss_matrix


def mst_decoder_old(chromosome: np.ndarray, diss_matrix: np.ndarray,
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


def mst_fitness_old(chromosome: np.ndarray, diss_matrix: np.ndarray,
                num_edges: int, K: int,
                G: igraph.Graph, diss_weights: np.ndarray,) -> float:
    P = mst_decoder_old(chromosome, diss_matrix, num_edges, K, G, diss_weights)
    return l2_objective_function_diss_matrix(P, diss_matrix)


# -------------------------------------------------------

import numpy as np

class UnionFind:
    def __init__(self, n):
        self._size = n
        self._parent = list(range(n))
        self._sizes = [1] * n

    def representative(self, id) -> int:
        "Find root of the tree to which id is connected"
        parent_id = self._parent[id]
        if  parent_id == id:
            return id 
        else:
            parent_repr = self.representative(parent_id)
            self._parent[id] = parent_repr # Path compression
            return parent_repr

    def connected(self, id1, id2) -> bool:
        "Are objects id1 and id2 connected?"
        return self.representative(id1) == self.representative(id2)

    def union(self, id1, id2):
        "Connect objects id1 and id2."
        root1 = self.representative(id1)
        tree1_size = self._sizes[root1]
        root2 = self.representative(id2)
        tree2_size = self._sizes[root2]
        # tree 1 is smaller, append to tree 2
        if tree1_size <= tree2_size:
            self._parent[root1] = root2
            self._sizes[root2] = tree1_size + tree2_size
        # tree 2 is smaller, append to tree 1
        else:
            self._parent[root2] = root1
            self._sizes[root1] = tree1_size + tree2_size
        

    def components(self) -> list[list[int]]:
        """
        Return a list of connected components.
        """
        roots = [False] * self._size
        components = [[k] for k in range(self._size)]
        for k in range(self._size):
            root_k = self.representative(k)
            if root_k != k:
                components[root_k].append(k)
            else:
                roots[k] = True
        return [components[k] for k in range(self._size) if roots[k]]   
    

def mst_decoder(chromosome: np.ndarray, diss_matrix: np.ndarray,
                num_edges: int, K: int,
                num_nodes: int,
                edges: np.ndarray,
                diss_weights: np.ndarray):
    
    # Get two weights from the chromosome 
    w_minus: np.ndarray = chromosome[:num_edges] * diss_weights
    w_plus: np.ndarray = chromosome[num_edges:]

    # Indicator of each edge in the MST
    mst_edges: np.ndarray = np.zeros(num_edges)
    edge_count: int = 0

    # Keep track of clusters while building the mst
    ds = UnionFind(num_nodes)

    # Greedy selection of the next edge
    edges_idx_order = np.lexsort( (np.arange(num_edges, dtype=np.int64), w_minus))
    for edge_idx in edges_idx_order:
        v1 = edges[edge_idx][0]
        v2 = edges[edge_idx][1]

        # Avoid cycles
        if ds.connected(v1, v2):
            continue

        # Select this edge
        mst_edges[edge_idx] = 1
        ds.union(v1, v2)
        edge_count += 1

        # Mst complete
        if edge_count == num_nodes - 1:
            break

    # Drop K-1 edges considering the weights in w_plus
    new_w_plus: np.ndarray = w_plus * mst_edges
    cut_edges: np.ndarray = np.argsort(new_w_plus, kind="mergesort")[-(K-1):]
    mst_edges[cut_edges] = 0

    # Construct a graph only with this edges
    ds = UnionFind(num_nodes)
    for edge_idx in range(num_edges):
        if mst_edges[edge_idx]:
            v1 = edges[edge_idx][0]
            v2 = edges[edge_idx][1]
            ds.union(v1, v2)

    # Make the partition using the connected components
    components = ds.components()
    P = {idx+1: c for idx, c in enumerate(components)}

    return P



def mst_fitness(chromosome: np.ndarray, diss_matrix: np.ndarray,
                num_edges: int, K: int,
                num_nodes: int,
                edges: np.ndarray,
                diss_weights: np.ndarray) -> float:
    P = mst_decoder(chromosome, diss_matrix, num_edges, K, num_nodes, edges, diss_weights)
    return l2_objective_function_diss_matrix(P, diss_matrix) # type: ignore







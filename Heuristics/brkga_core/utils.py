import numpy as np
import pandas as pd
import igraph
from typing import Callable, TypedDict
from numpy.typing import NDArray

# Basic types
Chromosome = NDArray[np.float64]
Offset = NDArray[np.int32]
Neighbors = NDArray[np.int32]
P_Array = NDArray[np.int32]
P_Dict = dict[int, list[int]]

# Important function types
Fit_Seq = Callable[[Chromosome, np.ndarray], float]
Fit_Par = Callable[[Chromosome, np.ndarray], float]
Decoder = Callable[[Chromosome, np.ndarray], P_Dict]


def build_adjacency_arrays(adjacency: dict, N: int) -> tuple[Offset, Neighbors]:
    """
    Transforms the adj dict into adj arrays.
    offsets - indicates what position in the array starts with the neighbors of a specific node
    neighbors - contains the neighbors from the first node, then second, etc
    Returns: (offsets, neighbors)
    """
    offsets = np.zeros(N + 1, dtype=np.int32) 
    neighbors = []
    
    current_offset = 0
    for i in range(N):
        offsets[i] = current_offset
        neigs = adjacency.get(i, [])
        neighbors.extend(neigs)
        current_offset += len(neigs)
    offsets[N] = current_offset
    
    return offsets, np.array(neighbors, dtype=np.int32)


def build_adjacency_arrays_with_edges(g: igraph.Graph) -> tuple[Offset, Neighbors, np.ndarray]:
    """
    Transforms the adj of the graph into three arrays
    offsets - indicates what position in the array starts with the neighbors of a specific node
    neighbors - contains the neighbors from the first node, then second, etc
    edges_id - contains the indiced of the edges following the neighbors array
    Returns: (offsets, neighbors, edge_idx)
    """
    offsets = np.zeros(g.vcount() + 1, dtype=np.int32) 
    neighbors = []
    edges_id = []
    
    current_offset = 0
    for i in range(g.vcount()):
        # set offset
        offsets[i] = current_offset
        # add neigbors and edge_idx
        neigs = []
        edges = []
        for v in g.neighbors(i):
            neigs.append(v)
            edges.append(g.get_eid(i, v))
        # update
        current_offset += len(neigs)
        neighbors.extend(neigs)
        edges_id.extend(edges)
    offsets[g.vcount()] = current_offset
    return offsets, np.array(neighbors, dtype=np.int32), np.array(edges_id, dtype=np.int32)




def P_from_array_to_dict(partition: P_Array, K: int) -> P_Dict:
    """  
    Go from the array representation to the dict representation
    """
    P = {idx: [] for idx in range(1, K+1)}
    for v, k in enumerate(partition):
        P[k+1].append(v)
    return P


class EvolutionStats(TypedDict):
    best_chromosome: Chromosome
    best_solution:  P_Dict
    best_fitness: float
    population_stats: pd.DataFrame
    time: float



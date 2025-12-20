import numpy as np
import pandas as pd
from numba import njit
from typing import Callable, TypedDict
from numpy.typing import NDArray

# Basic types
Chromosome = NDArray[np.float64]
Offset = NDArray[np.int32]
Neighbors = NDArray[np.int32]
N_k = NDArray[np.int32]
P_Array = NDArray[np.int32]
P_Dict = dict[int, list[int]]

# Important function types
Fit_Seq = Callable[[Chromosome, np.ndarray], float]
Fit_Par = Callable[[Chromosome, np.ndarray], float]
Decoder = Callable[[Chromosome, np.ndarray], P_Dict]


def build_adjacency_arrays(adjacency: dict, N: int) -> tuple[Offset, Neighbors]:
    """
    Transforms the adj dict into adj arrays.
    neighbors - contains the neighbors from the first node, then second, etc
    offsets - indicates what position in the array starts with the neighbors of a specific node
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


def P_from_array_to_dict(partition: P_Array, K: int) -> P_Dict:
    """  
    Go from the array representation to the dict representation
    """
    P = {idx: [] for idx in range(1, K+1)}
    for v, k in enumerate(partition):
        P[k+1].append(v)
    return P


@njit(fastmath=True, cache=True)
def l2_objective_array_version(N: int, K:int, partition: P_Array, n_k: N_k,
                               diss_matrix: np.ndarray) -> float:
    # Error of each region
    region_sums = np.zeros(K, dtype=np.float64)

    # Iterate on each pair of nodes in the same region
    for i in range(N):
        k = partition[i]
        for j in range(i + 1, N):
            if partition[j] == k:
                # Add the distance
                region_sums[k] += diss_matrix[i, j]

    # Add the total cost (divide each by n_k)
    total_cost = 0.0
    for k in range(K):
        total_cost += region_sums[k] / n_k[k]
    return total_cost


class EvolutionStats(TypedDict):
    best_chromosome: Chromosome
    best_solution:  P_Dict
    best_fitness: float
    population_stats: pd.DataFrame
    time: float



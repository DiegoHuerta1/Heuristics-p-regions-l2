import numpy as np
from numba import njit

from ..utils import l2_objective_array_version

@njit(cache=True)
def greedy_fitness(chromosome: np.ndarray, 
                    diss_matrix: np.ndarray,
                    N: int, K:int, rank: int, break_point: int,
                    adj_offsets: np.ndarray, adj_neighbors: np.ndarray) -> float:
    partition, n_k = greedy_decoder(chromosome, diss_matrix,
                                    N, K, rank, break_point, adj_offsets, adj_neighbors)
    return l2_objective_array_version(N, K, partition, n_k, diss_matrix)


@njit(fastmath=False, cache=True) 
def greedy_decoder(chromosome: np.ndarray, diss_matrix: np.ndarray,
                   N: int,  K: int, rank: int, break_point: int,
                   adj_offsets: np.ndarray, adj_neighbors: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Greedy Decoder
    """

    # Obtain the low rank matrix induced by the chromosome
    M = chromosome[:break_point].reshape((N, rank)).copy()   
    # Get K seed nodes from the second part (lowest values)
    seed_nodes: np.ndarray = np.argsort(chromosome[break_point:])[:K]
    
    # Start Partition with seeds  (0 index, -1 means unnasigned)
    partition: np.ndarray =  np.full(N, -1, dtype=np.int32)
    partition[seed_nodes] = np.arange(K)

    # Keep track of assigned nodes
    assigned_mask: np.ndarray = np.zeros(N, dtype=np.bool_)
    assigned_mask[seed_nodes] = True

    # Start R_k with zeros and n_k with ones
    R_k: np.ndarray = np.zeros(K, dtype=np.float64)
    n_k: np.ndarray = np.ones(K, dtype=np.int32)

    # Initialize the matrix that has the cost for each element (v, k)
    feasible_elements_g: np.ndarray = np.full((N, K), np.inf, dtype=np.float64)
    # Fill with initial feasible: {(v, k) | (∄h ∈ [K] : v ∈ Ph) ∧ (∃u ∈ N(v) : u ∈ Pk)}
    for u in seed_nodes:
        k = partition[u]
        for v in get_neighbors(adj_offsets, adj_neighbors, u):
            if not assigned_mask[v]:
                cost = get_lazy_dist(v, u, M, rank, diss_matrix)/2 # this is trivial
                feasible_elements_g[v, k] = cost

    # Create solution
    for _ in range(N - K):  # note that we must assign exactly N - K nodes

        # Get the element with lowest evaluation
        v_star = -1
        k_star = -1
        value_g = np.inf
        for v in range(N):
            for k in range(K):
                val = feasible_elements_g[v, k]
                if val < value_g:
                    value_g = val
                    v_star = v
                    k_star = k
        # Compute the future value of R_k_star after making the assignement
        future_R_k_star = R_k[k_star] + value_g

        # Remove elements that assign v_star to other regions
        for k in range(K):
            feasible_elements_g[v_star, k] = np.inf

        # Update greedy evaluations of elements that assign to k_star
        for v in range(N):
            old_eval = feasible_elements_g[v, k_star]
            if np.isfinite(old_eval):
                term1 = 1.0 / (n_k[k_star] + 2)
                term2 = (n_k[k_star] + 1)*old_eval 
                term3 = get_lazy_dist(v, v_star, M, rank, diss_matrix) - value_g
                new_eval = term1 * (term2 + term3)
                feasible_elements_g[v, k_star] = new_eval

        # Update the partition 
        partition[v_star] = k_star
        assigned_mask[v_star] = True
        R_k[k_star] = future_R_k_star
        n_k[k_star] += 1

        # Get new feasible elements and their evaluations
        # {(v, k∗) : (v ∈ N (v∗)) ∧ (∄h ∈ [K] : v ∈ Ph) ∧ ((v, k∗) /∈ F)}
        for v in get_neighbors(adj_offsets, adj_neighbors, v_star):
            if not assigned_mask[v] and np.isinf(feasible_elements_g[v, k_star]):
                sum_diss = sum_dissimilarities(v, k_star, M, rank, diss_matrix, partition, N)
                cost = 1/(n_k[k_star] + 1) * (sum_diss - R_k[k_star])
                feasible_elements_g[v, k_star] = cost

    return partition, n_k


# --------------------------------------------------------
# HELPER FUNCTIONS
    
@njit(cache=True)
def get_lazy_dist(i: int, j: int, M: np.ndarray, rank: int, diss_matrix: np.ndarray) -> float:
    dot_prod = 0.0
    for k in range(rank):
        dot_prod += M[i, k] * M[j, k]
    return dot_prod * diss_matrix[i, j]

@njit(cache=True)
def get_neighbors(adj_offset: np.ndarray, adj_neighbors: np.ndarray, v: int) -> np.ndarray:
    start = adj_offset[v]
    end = adj_offset[v + 1]
    return adj_neighbors[start:end]

@njit(cache=True)
def sum_dissimilarities(v: int, k: int, 
                        M: np.ndarray, rank: int, diss_matrix: np.ndarray,
                        partition: np.ndarray, N: int,) -> float:
    # sum(matrix_d[v, i] for i in P[k])
    sum_diss = 0.0
    for i in range(N):
        if partition[i] == k:
            dist = get_lazy_dist(v, i, M, rank, diss_matrix)
            sum_diss += dist
    return sum_diss






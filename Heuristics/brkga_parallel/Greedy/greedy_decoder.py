import numpy as np
import itertools

from ...utils import l2_objective_function_diss_matrix

# --------------------------------------------------------
# COMPUTE FITNESS 

def chromosome_fitness(chromosome: np.ndarray, 
                dissimilarity_matrix: np.ndarray,
                N: int, rank: int, break_point: int,
                K: int,
                adjacency: dict[int, list[int]]) -> float:
    solution = decode(chromosome, dissimilarity_matrix, N, rank, break_point, K, adjacency)
    return l2_objective_function_diss_matrix(solution, dissimilarity_matrix)


# --------------------------------------------------------
# DECODE

def decode(chromosome: np.ndarray, 
           dissimilarity_matrix: np.ndarray,
           N: int, rank: int, break_point: int,
           K: int,
           adjacency: dict[int, list[int]]) -> dict[int, list[int]]:
    """
    Decode a chromosome into a solution.
    
    Greedy Decoder
    """

    # Obtain the dissimilarit matrix induced by the chromosome
    matrix_c = get_matrix_from_chromosome(chromosome[:break_point], N, rank)
    # matrix_d = matrix_c * dissimilarity_matrix
    matrix_d =  matrix_c * dissimilarity_matrix
    
    # Get K seed nodes from the second part (lowest values)
    seed_nodes = np.argsort(chromosome[break_point:])[:K]
    
    # Keep track of assigned nodes
    assigned_nodes: set[int] = set(seed_nodes)
    # Start Partition with seeds 
    P: dict[int, list[int]] = {(idx+1): [int(seed)] for idx, seed in enumerate(seed_nodes)}
    # Start R_k with zeros
    R_k: dict[int, float] = {k: 0.0 for k in P.keys()}

    # Get feasible elements, and their evaluation under the greedy function
    feasible_elements_g: dict[tuple[int, int], float] = get_feasible_elements_greedy(P, 
                                                                                     adjacency,
                                                                                     assigned_nodes,
                                                                                     matrix_d,
                                                                                     R_k)

    # Create solution while there are feasible elements
    while feasible_elements_g:

        # Get the element with lowest evaluation
        v_star, k_star = min(feasible_elements_g, key=lambda e: feasible_elements_g[e])

        # Compute the future value of R_k_star after making the assignement
        future_R_k_star = compute_future_R_k(v_star, k_star, matrix_d, P, R_k)

        # Remove elements that assign v_star to other regions
        feasible_elements_g = {e: val for e, val in feasible_elements_g.items() if e[0] != v_star}
        # Update greedy evaluations of elements that assign to k_star
        for (v, k) in feasible_elements_g.keys():
            if k == k_star:
                feasible_elements_g[(v, k)] = update_greedy_eval(v, k, feasible_elements_g[(v, k)],
                                                                        v_star, future_R_k_star,
                                                                        matrix_d, P, R_k)

        # Update the partition and R_k
        P[k_star].append(v_star)
        R_k[k_star] = future_R_k_star
        assigned_nodes.add(v_star)

        # Get new feasible elements and their evaluations
        new_feasible_elements_g = get_new_feasible_elements_greedy(v_star, k_star,
                                                                    feasible_elements_g.keys(),
                                                                    matrix_d,
                                                                    adjacency, assigned_nodes,
                                                                    P, R_k)
        feasible_elements_g.update(new_feasible_elements_g)

    return P



# --------------------------------------------------------
# HELPER FUNCTIONS
    

def get_matrix_from_chromosome(vector, N: int, r: int):
    M = np.reshape(vector, (N, r))
    G = M @ M.T
    return G


def vec_to_sym(vector: np.ndarray, N : int) -> np.ndarray:
    matrix = np.zeros((N, N))
    upper_indices = np.triu_indices(N, k=1)
    matrix[upper_indices] = vector
    matrix += matrix.T
    return matrix



def get_feasible_elements_greedy(P: dict[int, list[int]],
                                adjacency: dict[int, list[int]],
                                assigned_nodes: set[int],  
                                matrix_d: np.ndarray,
                                R_k: dict[int, float]
                                ) -> dict[tuple[int, int], float]:
    # Compute all feasible elements 
    # {(v, k) | (∄h ∈ [K] : v ∈ Ph) ∧ (∃u ∈ N(v) : u ∈ Pk)}
    feasible_elements: list[tuple] = []
    # iterate on asigned nodes
    for k, P_k in P.items():
        for u in P_k:
            # iterate on unnasigned neighbors
            for v in adjacency[u]:
                if v not in assigned_nodes:
                    # save the element
                    feasible_elements.append((v, k))

    # Evaluate all feasible elements under the greedy function
    feasible_elements_g: dict[tuple[int, int], float] = {}
    for (v, k) in feasible_elements:
        feasible_elements_g[(v, k)] = evaluate_greedy_element(v, k, matrix_d, P, R_k)
    return feasible_elements_g

def evaluate_greedy_element(v: int, k: int, matrix_d: np.ndarray,
                            P: dict[int, list[int]], R_k: dict[int, float]) -> float:
    sum_dissimilarities = sum(matrix_d[v, i] for i in P[k])
    evaluation = 1/(len(P[k]) + 1) * (sum_dissimilarities - R_k[k])
    return evaluation

def update_greedy_eval(v: int, k: int, old_eval: float, 
                        v_star: int, new_R_k: float,
                        matrix_d: np.ndarray,
                        P: dict[int, list[int]], R_k: dict[int, float]) -> float:
    n_k = len(P[k])
    new_eval = 1/(n_k + 2) * ((n_k + 1)*old_eval + R_k[k] - new_R_k + matrix_d[v, v_star])
    return new_eval

def compute_future_R_k(v: int, k: int, matrix_d: np.ndarray,
                       P: dict[int, list[int]], R_k: dict[int, float]) -> float:
    n_k = len(P[k])
    return 1/(n_k + 1) * (n_k * R_k[k] + sum(matrix_d[v, i] for i in P[k]))

def get_new_feasible_elements_greedy(v_star: int, k_star: int,
                                     current_feasible, matrix_d: np.ndarray,
                                     adjacency: dict[int, list[int]], assigned_nodes: set[int],
                                     P: dict[int, list[int]], R_k: dict[int, float]) -> dict[tuple[int, int], float]:
    # Compute feasible elements 
    # {(v, k∗) : (v ∈ N (v∗)) ∧ (∄h ∈ [K] : v ∈ Ph) ∧ ((v, k∗) /∈ F)}
    new_feasible_elements: list[tuple] = []
    for v in adjacency[v_star]:
        if v not in assigned_nodes and (v, k_star) not in current_feasible:
            new_feasible_elements.append((v, k_star))
    # Evaluate all new feasible elements under the greedy function
    new_feasible_elements_g: dict[tuple[int, int], float] = {}
    for (v, k) in new_feasible_elements:
        new_feasible_elements_g[(v, k)] = evaluate_greedy_element(v, k, matrix_d, P, R_k)
    return new_feasible_elements_g




import numpy as np
from numba import njit


# Union Find ---------------------------------

@njit(cache=True)
def ds_find_root(ds_parents: np.ndarray, x: int) -> int:
    """ Find the root of x""" 
    parent = ds_parents[x]
    while parent != x:
        x = parent
        parent = ds_parents[x]
    return x

@njit(cache=True)
def ds_find_root_pc(ds_parents: np.ndarray, x: int) -> int:
    """ Find the root of x and use path compression""" 
    parent = ds_parents[x]
    while parent != x:
        ds_parents[x] = ds_parents[parent] 
        x = parent
        parent = ds_parents[x]
    return x


@njit(cache=True)
def ds_get_roots(ds_parents: np.ndarray, size: int) -> np.ndarray:
    """ Get an array with all the roots """
    roots = ds_parents.copy()
    for x in range(size):
        if x != roots[x]:
            roots[x] = ds_find_root(ds_parents, x)
    return roots    


# Main function ----------------------------

@njit(cache=True)
def msf_almost_decoder(chromosome: np.ndarray,
                       num_edges: int, K: int,
                       num_nodes: int,
                       edges: np.ndarray,
                       diss_weights: np.ndarray) -> np.ndarray:
    """  
    Get the roots array from the chromosome
    """
    
    # Get edge weigths from the first part
    w: np.ndarray = chromosome[:num_edges] * diss_weights

    # Get K seed nodes from the second part (lowest values)
    x: np.ndarray = chromosome[num_edges:]
    order: np.ndarray = np.argsort(x)
    seed_nodes: np.ndarray = order[:K]
    is_seed: np.ndarray = np.zeros(num_nodes, dtype=np.bool_)
    is_seed[seed_nodes] = True

    # # Delete
    # for edge_idx in range(num_edges):
    #     v1: int = edges[edge_idx][0]
    #     v2: int = edges[edge_idx][1]
    #     if is_seed[v1] or is_seed[v2]:
    #         w[edge_idx] /= 2

    # Indicator of each edge in the MSF
    msf_edges: np.ndarray = np.zeros(num_edges)
    edge_count: int = 0

    # Keep track of clusters while building the mst
    ds_parents: np.ndarray = np.arange(num_nodes, dtype=np.int32)
    ds_sizes: np.ndarray = np.ones(num_nodes, dtype=np.int32)

    # Greedy selection of the next edge (v1, v2)
    edges_idx_order = np.argsort(w)
    for edge_idx in edges_idx_order:
        v1: int = edges[edge_idx][0]
        v2: int = edges[edge_idx][1]

        # Get roots and tree sizes
        r1: int = ds_find_root_pc(ds_parents, v1)
        r2: int = ds_find_root_pc(ds_parents, v2)
        tree1_size: int = ds_sizes[r1]
        tree2_size: int = ds_sizes[r2]

        # Avoid cycles
        if r1 == r2:
            continue
        # Avoid connecting two seeds
        if is_seed[r1] and is_seed[r2]:
            continue

        # The seed nodes are always roots
        if is_seed[r1]:
            ds_parents[r2] = r1
            ds_sizes[r1] +=  tree2_size
        elif is_seed[r2]:
            ds_parents[r1] = r2
            ds_sizes[r2] += tree1_size

        # They are not seeds, use size to decide
        else:
            if tree2_size <= tree1_size:
                ds_parents[r2] = r1
                ds_sizes[r1] +=  tree2_size
            else:
                ds_parents[r1] = r2
                ds_sizes[r2] += tree1_size

        # Mark edge as complete
        msf_edges[edge_idx] = 1
        edge_count += 1
        # Msf complete (K trees)
        if edge_count == num_nodes - K:
            break


    # Get the roots of the connected components
    return ds_get_roots(ds_parents, num_nodes)


# Fitness -------------------------------------

@njit(fastmath=True, cache=True)
def l2_objective_from_roots(roots: np.ndarray, num_nodes: int, K:int,
                            diss_matrix: np.ndarray) -> float:
    region_sums: np.ndarray = np.zeros(num_nodes, dtype=np.float64)
    region_counts: np.ndarray = np.zeros(num_nodes, dtype=np.int32)

    # Iterate on each pair of nodes 
    for i in range(num_nodes):
        r = roots[i]
        # Add the size of the region
        region_counts[r] += 1
        for j in range(i + 1, num_nodes):
            if roots[j] == r:
                # Add the distance
                region_sums[r] += diss_matrix[i, j]

    # Add the total cost (divide each by the size)
    total_cost = 0.0
    for r in range(num_nodes):
        if region_counts[r] > 0:
            total_cost += region_sums[r] / region_counts[r]
    return total_cost

@njit(cache=True)
def msf_fitness(chromosome: np.ndarray, diss_matrix: np.ndarray,
                num_edges: int, K: int,
                num_nodes: int,
                edges: np.ndarray,
                diss_weights: np.ndarray) -> float:
    roots_array: np.ndarray = msf_almost_decoder(chromosome,
                                                 num_edges, K, num_nodes,
                                                 edges, diss_weights)
    
    return l2_objective_from_roots(roots_array, num_nodes, K, diss_matrix)


# Decoder --------------------------------

@njit(cache=True)
def relabel_components(roots: np.ndarray, num_nodes: int) -> np.ndarray:
    """  
    Relabel roots (arbitrary indices) to labels 0, 1, ...
    """
    mapping: np.ndarray = np.full(num_nodes, -1) # map from root to label
    next_label = 0
    labels = np.empty(num_nodes, dtype=np.int32)

    # Iterate on each node
    for i in range(num_nodes):
        r = roots[i]
        # Possibly add a new label
        if mapping[r] == -1: 
            mapping[r] = next_label
            next_label += 1
        # relabel root 
        labels[i] = mapping[r]

    return labels

@njit(cache=True)
def msf_decoder(chromosome: np.ndarray, 
                num_edges: int, K: int,
                num_nodes: int,
                edges: np.ndarray,
                diss_weights: np.ndarray) -> np.ndarray:
    
    roots_array: np.ndarray = msf_almost_decoder(chromosome,
                                                 num_edges, K, num_nodes,
                                                 edges, diss_weights)
    return relabel_components(roots_array, num_nodes)




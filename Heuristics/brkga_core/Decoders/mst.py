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
def ds_connected(ds_parents: np.ndarray, x1: int, x2: int) -> bool:
    """ Check if x1 and x2 are in the same component """
    return ds_find_root_pc(ds_parents, x1) == ds_find_root_pc(ds_parents, x2)

@njit(cache=True)
def ds_union(ds_parents: np.ndarray, ds_sizes: np.ndarray, x1: int, x2:int):
    """ Connect x1 and x2 """
    root1 = ds_find_root_pc(ds_parents, x1)
    root2 = ds_find_root_pc(ds_parents, x2)
    if root1 == root2:
        return
    tree1_size = ds_sizes[root1]
    tree2_size = ds_sizes[root2]
    # tree 1 is smaller, append to tree 2
    if tree1_size <= tree2_size:
        ds_parents[root1] = root2
        ds_sizes[root2] += tree1_size
    # tree 2 is smaller, append to tree 1
    else:
        ds_parents[root2] = root1
        ds_sizes[root1] +=  tree2_size

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
def mst_almost_decoder(chromosome: np.ndarray,
                       num_edges: int, K: int,
                       num_nodes: int,
                       edges: np.ndarray,
                       diss_weights: np.ndarray) -> np.ndarray:
    """  
    Get the roots array from the chromosome
    """
    
    # Get two weights from the chromosome 
    w_minus: np.ndarray = chromosome[:num_edges] * diss_weights
    w_plus: np.ndarray = chromosome[num_edges:]

    # Indicator of each edge in the MST
    mst_edges: np.ndarray = np.zeros(num_edges)
    edge_count: int = 0

    # Keep track of clusters while building the mst
    ds_parents: np.ndarray = np.arange(num_nodes, dtype=np.int32)
    ds_sizes: np.ndarray = np.ones(num_nodes, dtype=np.int32)

    # Greedy selection of the next edge
    edges_idx_order = np.argsort(w_minus, kind="mergesort") # mergesort performs lex sort
    for edge_idx in edges_idx_order:
        v1 = edges[edge_idx][0]
        v2 = edges[edge_idx][1]

        # Avoid cycles
        if ds_connected(ds_parents, v1, v2):
            continue

        # Select this edge
        mst_edges[edge_idx] = 1
        ds_union(ds_parents, ds_sizes, v1, v2)
        edge_count += 1
        # Mst complete
        if edge_count == num_nodes - 1:
            break

    # Drop K-1 edges considering the weights in w_plus
    new_w_plus: np.ndarray = w_plus * mst_edges
    cut_edges: np.ndarray = np.argsort(new_w_plus, kind="mergesort")[-(K-1):]
    mst_edges[cut_edges] = 0

    # Construct a graph only with this edges
    ds_parents: np.ndarray = np.arange(num_nodes, dtype=np.int32)
    ds_sizes: np.ndarray = np.ones(num_nodes, dtype=np.int32)
    for edge_idx in range(num_edges):
        if mst_edges[edge_idx]:
            v1 = edges[edge_idx][0]
            v2 = edges[edge_idx][1]
            ds_union(ds_parents, ds_sizes, v1, v2)

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
def mst_fitness(chromosome: np.ndarray, diss_matrix: np.ndarray,
                num_edges: int, K: int,
                num_nodes: int,
                edges: np.ndarray,
                diss_weights: np.ndarray) -> float:
    roots_array: np.ndarray = mst_almost_decoder(chromosome,
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
def mst_decoder(chromosome: np.ndarray, 
                num_edges: int, K: int,
                num_nodes: int,
                edges: np.ndarray,
                diss_weights: np.ndarray) -> np.ndarray:
    
    roots_array: np.ndarray = mst_almost_decoder(chromosome,
                                                 num_edges, K, num_nodes,
                                                 edges, diss_weights)
    return relabel_components(roots_array, num_nodes)




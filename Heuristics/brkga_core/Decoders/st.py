import numpy as np

# Dijkstra -------------------------------------

import heapq
def multi_source_dijkstra(seed_nodes: np.ndarray, w: np.ndarray,
                           num_nodes: int, diss_matrix: np.ndarray,
                          adj_offsets: np.ndarray, adj_neighbors: np.ndarray,
                          adj_edges: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        
    # Keep track of min dist and the seed that leads to that minimum
    min_dist: np.ndarray = np.full(num_nodes, np.inf, dtype=np.float64)
    seed: np.ndarray = np.full(num_nodes, -1, dtype=np.int32)

    # Initialize priority queue empty
    pq = []

    # Initialize each seed node
    for s in seed_nodes:
        min_dist[s] = 0.0
        seed[s] = s
        heapq.heappush(pq, (0.0, s))

    # Pop an element
    while len(pq) > 0:
        dist_u, u = heapq.heappop(pq)

        # if we already computed a better path
        if min_dist[u] < dist_u:
            continue

        # for each neighbor v of u
        for idx in range(adj_offsets[u], adj_offsets[u + 1]):
            v = adj_neighbors[idx]
            edge_id = adj_edges[idx]

            # check if we can improve the distance
            new_dist_v = dist_u + w[edge_id]
            if (new_dist_v < min_dist[v]) or (new_dist_v == min_dist[v] and diss_matrix[seed[u], v] < diss_matrix[seed[v], v]):
                min_dist[v] = new_dist_v
                seed[v] = seed[u]
                # push 
                heapq.heappush(pq, (new_dist_v, v))

    return min_dist, seed


# Main function -------------------------------


def st_almost_decoder(chromosome: np.ndarray, diss_matrix: np.ndarray,
                      num_nodes: int, num_edges: int, K: int,
                      diss_weights: np.ndarray, adj_offsets: np.ndarray,
                      adj_neighbors: np.ndarray, adj_edges: np.ndarray) -> np.ndarray:
    
    # Get edge weigths from the first part
    w = chromosome[:num_edges] * diss_weights

    # Get K seed nodes from the second part (lowest values)
    x = chromosome[num_edges:]
    order = np.argsort(x, kind="mergesort")
    seed_nodes = order[:K]

    # Run multi source dijkstra
    _, seeds = multi_source_dijkstra(seed_nodes,w, num_nodes,
                                    diss_matrix, adj_offsets, 
                                    adj_neighbors, adj_edges)
    return seeds


# Fitness -------------------------------------

def l2_objective_from_seeds(seeds: np.ndarray, num_nodes: int,
                            diss_matrix: np.ndarray) -> float:
    
    region_sums: np.ndarray = np.zeros(num_nodes, dtype=np.float64)
    region_counts: np.ndarray = np.zeros(num_nodes, dtype=np.int32)

    # Iterate on each pair of nodes 
    for i in range(num_nodes):
        s = seeds[i]
        # Add the size of the region
        region_counts[s] += 1
        for j in range(i + 1, num_nodes):
            if seeds[j] == s:
                # Add the distance
                region_sums[s] += diss_matrix[i, j]

    # Add the total cost (divide each by the size)
    total_cost = 0.0
    for r in range(num_nodes):
        if region_counts[r] > 0:
            total_cost += region_sums[r] / region_counts[r]
    return total_cost


def st_fitness(chromosome: np.ndarray, diss_matrix: np.ndarray,
               num_nodes: int, num_edges: int, K: int,
               diss_weights: np.ndarray, adj_offsets: np.ndarray,
               adj_neighbors: np.ndarray, adj_edges: np.ndarray) -> float:
    seeds_array = st_almost_decoder(chromosome, diss_matrix,
                                    num_nodes, num_edges, K,
                                    diss_weights, adj_offsets,
                                    adj_neighbors, adj_edges)
    return l2_objective_from_seeds(seeds_array, num_nodes, diss_matrix)


# Decoder --------------------------------


def relabel_components(seeds: np.ndarray, num_nodes: int) -> np.ndarray:
    """  
    Relabel seeds (arbitrary indices) to labels 0, 1, ...
    """
    mapping: np.ndarray = np.full(num_nodes, -1) # map from seeds to label
    next_label = 0
    labels = np.empty(num_nodes, dtype=np.int32)

    # Iterate on each node
    for i in range(num_nodes):
        s = seeds[i]
        # Possibly add a new label
        if mapping[s] == -1: 
            mapping[s] = next_label
            next_label += 1
        # relabel root 
        labels[i] = mapping[s]

    return labels


def st_decoder(chromosome: np.ndarray, diss_matrix: np.ndarray,
               num_nodes: int, num_edges: int, K: int,
               diss_weights: np.ndarray, adj_offsets: np.ndarray,
               adj_neighbors: np.ndarray, adj_edges: np.ndarray) -> np.ndarray:
    seeds_array = st_almost_decoder(chromosome, diss_matrix,
                                    num_nodes, num_edges, K,
                                    diss_weights, adj_offsets,
                                    adj_neighbors, adj_edges)
    return relabel_components(seeds_array, num_nodes)

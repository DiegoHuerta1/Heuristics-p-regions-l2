import numpy as np

# Binary Heap --------------------------
# The root element will be at arr[0]
# arr[(i-1)//2]	Returns the parent node
# arr[(2*i)+1]	Returns the left child node
# arr[(2*i)+2]	Returns the right child node

def heap_push(heap_dist: np.ndarray, heap_node: np.ndarray, heap_size: int,
              dist_v: float, v: int) -> int:
    # start at the end
    idx_new = heap_size
    parent_index = (idx_new - 1) // 2

    # Climb if the parent is greater
    while idx_new > 0 and heap_dist[(idx_new - 1)//2] > dist_v:
        heap_dist[idx_new] = heap_dist[parent_index]
        heap_node[idx_new] = heap_node[parent_index]
        # update index
        idx_new = parent_index
        parent_index = (idx_new - 1) // 2

    # insert the new value at this index
    heap_dist[idx_new] = dist_v
    heap_node[idx_new] = v

    return heap_size + 1


def heap_pop(heap_dist: np.ndarray, heap_node: np.ndarray,
            heap_size: int) -> tuple[float, int, int]:
    # Retrieve the min value
    min_value = heap_dist[0]
    u = heap_node[0]

    # Replace the root of the heap with the last element
    heap_size = heap_size - 1
    last_dist = heap_dist[heap_size]
    last_node = heap_node[heap_size]

    # Start at the begining
    idx: int = 0
    left_child_idx: int = 2 * idx + 1
    right_child_idx: int 
    child_idx: int


    # if there are valid childs
    while left_child_idx < heap_size:
        right_child_idx = left_child_idx + 1

        # select right
        if right_child_idx < heap_size and heap_dist[right_child_idx] < heap_dist[left_child_idx]:
            child_idx = right_child_idx
        # select left
        else:
            child_idx = left_child_idx

        # check if we do need to move
        if last_dist <= heap_dist[child_idx]:
            break

        # move to the child
        heap_dist[idx] = heap_dist[child_idx]
        heap_node[idx] = heap_node[child_idx]
        idx = child_idx
        left_child_idx = 2 * idx + 1
            
    # Insert the last element in the correct position
    heap_dist[idx] = last_dist
    heap_node[idx] = last_node

    return min_value, u, heap_size



# Dijkstra -------------------------------------

def multi_source_dijkstra(seed_nodes: np.ndarray, w: np.ndarray,
                           num_nodes: int,  num_edges: int, diss_matrix: np.ndarray,
                          adj_offsets: np.ndarray, adj_neighbors: np.ndarray,
                          adj_edges: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    # Keep track of min dist and the seed that leads to that minimum
    min_dist: np.ndarray = np.full(num_nodes, np.inf, dtype=np.float64)
    seed: np.ndarray = np.full(num_nodes, -1, dtype=np.int32)

    # Initialize binary heap for pairs (dist_v, v)
    max_heap_size: int = len(adj_neighbors) # num_edges
    heap_dist: np.ndarray = np.empty(max_heap_size, dtype=np.float64)
    heap_node: np.ndarray = np.empty(max_heap_size, dtype=np.int32)
    heap_size: int = 0

    # Initialize each seed node
    for s in seed_nodes:
        min_dist[s] = 0.0
        seed[s] = s
        # Push in heap: heapq.heappush(pq, (0.0, s))
        heap_size = heap_push(heap_dist, heap_node, heap_size, 0.0, s)

    while heap_size > 0:
        # Pop an element: dist_u, u = heapq.heappop(pq)
        dist_u, u, heap_size = heap_pop(heap_dist, heap_node, heap_size)        

        # if we already computed a better path
        if min_dist[u] < dist_u:
            continue

        # for each neighbor v of u
        for idx in range(adj_offsets[u], adj_offsets[u + 1]):
            v = adj_neighbors[idx]
            edge_id: int = adj_edges[idx]

            # check if we can improve the distance
            new_dist_v = dist_u + w[edge_id]
            if (new_dist_v < min_dist[v]) or (new_dist_v == min_dist[v] and seed[u] < seed[v]):
                min_dist[v] = new_dist_v
                seed[v] = seed[u]
                # push: heapq.heappush(pq, (new_dist_v, v))
                heap_size = heap_push(heap_dist, heap_node, heap_size, new_dist_v, v)

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
                                     num_edges, diss_matrix,
                                     adj_offsets, adj_neighbors,
                                     adj_edges)
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

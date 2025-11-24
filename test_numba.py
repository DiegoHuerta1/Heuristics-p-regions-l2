from Heuristics.utils import generate_dissimilarity_matrix
import igraph
import os
import time
import numpy as np


def main():   
 
    # Define instance
    instance_path = "./Instances_Mexico/12.pkl"
    plots_path = "./Example_plots/"
    os.makedirs(plots_path, exist_ok=True)
    num_regions = 10
    # Reproducibility
    seed = 0
    
    # Read instance
    with open(instance_path, "rb") as f:
        graph = igraph.Graph.Read_Pickle(f)
    print(f"Instance with {graph.vcount()} nodes")
    print(f"Regionalization in {num_regions} regions\n")
    
    # Compute dissimilarity matrix
    diss_matrix = generate_dissimilarity_matrix(graph)
    
    # BRKGA parameters
    config = {
        "population_size": 100,
        "elite_fraction": 0.2,
        "mutant_fraction": 0.2,
        "crossover_rate": 0.7,
        "max_generations": 1,
        "tolerance_generations": 100,
        "max_time": 9000,  
        "seed": seed,
    }

    # Load both classes
    from Heuristics.brkga_parallel.test_brkga_parallel import Greedy_BRKGA_parallel_test
    brkga = Greedy_BRKGA_parallel_test(graph, num_regions, diss_matrix,
                                        rank = 1, verbose = False, num_workers = 4,
                                        **config)
    from Heuristics.brkga_parallel.test_brkga_parallel_n import Greedy_BRKGA_parallel_test_n
    brkga_n = Greedy_BRKGA_parallel_test_n(graph, num_regions, diss_matrix,
                                         rank = 1, verbose = False, num_workers = 4,
                                         **config)
    num = 1000
    X = brkga.generate_chromosome_array(num)

    # OLD Implementation (no numba)
    from Heuristics.brkga_parallel.test_decoder import l2_objective_function_diss_matrix, decode
    costs_old = []
    start_time = time.time()
    for c in X:
        P = decode(c, brkga.dissimilarity_matrix, brkga.N,
                   brkga.rank, brkga.break_point, brkga.K,
                   brkga.adjacency)
        cost = l2_objective_function_diss_matrix(P, brkga.dissimilarity_matrix)
        costs_old.append(cost)
    total_time = time.time() - start_time
    print(f"Old implementation, time: {total_time}")

    # NEW Implementation (numba)
    from Heuristics.brkga_parallel.test_decoder_n import l2_objective_array_version, decode_n
    costs_new = []
    # warm up
    start_time = time.time()
    c = X[0]
    p, n_k = decode_n(c, brkga_n.dissimilarity_matrix, brkga_n.N,
                             brkga_n.rank, brkga_n.break_point, brkga_n.K,
                             brkga_n.adj_offsets, brkga_n.adj_neighbors)
    l2_objective_array_version(brkga_n.N, brkga_n.K, p, n_k, brkga_n.dissimilarity_matrix)
    total_time = time.time() - start_time
    print(f"Warmup time: {total_time}")
    # testing new
    start_time = time.time()
    for c in X:
        p, n_k = decode_n(c, brkga_n.dissimilarity_matrix, brkga_n.N,
                             brkga_n.rank, brkga_n.break_point, brkga_n.K,
                             brkga_n.adj_offsets, brkga_n.adj_neighbors)
        cost = l2_objective_array_version(brkga_n.N, brkga_n.K, p, n_k, brkga_n.dissimilarity_matrix)
        costs_new.append(cost)
    total_time = time.time() - start_time
    print(f"New implementation, time: {total_time}")

    # Compare results
    costs_old = np.array(costs_old)
    costs_new = np.array(costs_new)
    print("Testing equality")
    print(np.all(np.isclose(costs_old, costs_new)))
    print(costs_old.mean())
    print(costs_new.mean())
    #import matplotlib.pyplot as plt
    #plt.boxplot([costs_old, costs_new], tick_labels=["old", "new"])
    #plt.tight_layout()
    #plt.show()


if __name__ == "__main__":
    main()





    
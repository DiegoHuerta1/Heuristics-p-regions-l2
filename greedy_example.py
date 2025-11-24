from Heuristics.utils import generate_dissimilarity_matrix
import igraph
import os
import time


def main():   
 
    # Define instance
    instance_path = "./Instances_Mexico/07.pkl"
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
        "population_size": 500,
        "elite_fraction": 0.2,
        "mutant_fraction": 0.2,
        "crossover_rate": 0.7,
        "max_generations": 400,
        "tolerance_generations": 100,
        "max_time": 9000,  
        "seed": seed,
    }

    # Secuential code (functional version)
    # print("-"*100)
    # print("Functional\n")
    # from Heuristics.brkga_parallel.brkga_functional import Greedy_BRKGA_functional
    # brkga = Greedy_BRKGA_functional(graph, num_regions, diss_matrix, **config)
    # brkga.run()
    # brkga.print_statistics()
    # brkga.plot_evolution(plots_path + "greedy_func_evolution.png") 

    # -----------------------------------------------------------------------
    # New functional (parallel, + low rank)
    # print("-"*100)
    # print("Functional (new)\n")
    # from Heuristics.brkga_parallel.test_brkga_functional import Greedy_BRKGA_functional_test
    # brkga = Greedy_BRKGA_functional_test(graph, num_regions, diss_matrix,
    #                                      rank = 1, verbose = True, **config)
    # brkga.run()
    # brkga.print_statistics()
    # brkga.plot_evolution(plots_path + "greedy_func_test_evolution.png") 
    # # Compare with a chromosome with the same seeds but constant
    # from Heuristics.brkga_parallel.test_decoder import chromosome_fitness
    # best_c = brkga.evolution_stats["best_chromosome"]
    # best_c_seeds = best_c.copy()
    # best_c_seeds[:brkga.break_point] = 1 # identity multiplicative
    # f_original = chromosome_fitness(best_c,
    #                                brkga.dissimilarity_matrix,
    #                                brkga.N,
    #                                brkga.rank,
    #                                brkga.break_point,
    #                                brkga.K,
    #                                brkga.adjacency)
    # f_compare = chromosome_fitness(best_c_seeds,
    #                                brkga.dissimilarity_matrix,
    #                                brkga.N,
    #                                brkga.rank,
    #                                brkga.break_point,
    #                                brkga.K,
    #                                brkga.adjacency)
    # print(f"Fitness of best chromosome: {f_original}")
    # print(f"Fitness of chromosome with same seeds: {f_compare}\n")


    # -----------------------------------------------------------------------
    # Test Parallel (low rank)
    # print("-"*100)
    # print("Parallel (with shared memory)\n")
    # from Heuristics.brkga_parallel.test_brkga_parallel import Greedy_BRKGA_parallel_test
    # brkga = Greedy_BRKGA_parallel_test(graph, num_regions, diss_matrix,
    #                                    rank = 1, verbose = False, num_workers = 4,
    #                                     **config)
    # brkga.run()
    # brkga.print_statistics()
    # brkga.plot_evolution(plots_path + "greedy_func_test_evolution.png") 
    # # Compare with a chromosome with the same seeds but constant
    # from Heuristics.brkga_parallel.test_decoder import chromosome_fitness
    # best_c = brkga.evolution_stats["best_chromosome"]
    # best_c_seeds = best_c.copy()
    # best_c_seeds[:brkga.break_point] = 1 # identity multiplicative
    # f_original = chromosome_fitness(best_c,
    #                                brkga.dissimilarity_matrix,
    #                                brkga.N,
    #                                brkga.rank,
    #                                brkga.break_point,
    #                                brkga.K,
    #                                brkga.adjacency)
    # f_compare = chromosome_fitness(best_c_seeds,
    #                                brkga.dissimilarity_matrix,
    #                                brkga.N,
    #                                brkga.rank,
    #                                brkga.break_point,
    #                                brkga.K,
    #                                brkga.adjacency)
    # print(f"Fitness of best chromosome: {f_original}")
    # print(f"Fitness of chromosome with same seeds: {f_compare}\n")


    # -----------------------------------------------------------------------
    # New Parallel (low rank + numba)
    print("-"*100)
    print("Parallel (with shared memory and numba!)\n")
    from Heuristics.brkga_parallel.test_brkga_parallel_n import Greedy_BRKGA_parallel_test_n
    brkga = Greedy_BRKGA_parallel_test_n(graph, num_regions, diss_matrix,
                                       rank = 1, verbose = False, num_workers = 4,
                                        **config)
    brkga.run()
    brkga.print_statistics()
    brkga.plot_evolution(plots_path + "greedy_func_test_evolution.png") 
    # Compare with a chromosome with the same seeds but constant
    from Heuristics.brkga_parallel.test_decoder import chromosome_fitness
    best_c = brkga.evolution_stats["best_chromosome"]
    best_c_seeds = best_c.copy()
    best_c_seeds[:brkga.break_point] = 1 # identity multiplicative
    f_original = chromosome_fitness(best_c,
                                   brkga.dissimilarity_matrix,
                                   brkga.N,
                                   brkga.rank,
                                   brkga.break_point,
                                   brkga.K,
                                   brkga.adjacency)
    f_compare = chromosome_fitness(best_c_seeds,
                                   brkga.dissimilarity_matrix,
                                   brkga.N,
                                   brkga.rank,
                                   brkga.break_point,
                                   brkga.K,
                                   brkga.adjacency)
    print(f"Fitness of best chromosome: {f_original}")
    print(f"Fitness of chromosome with same seeds: {f_compare}\n")



if __name__ == "__main__":
    from cProfile import Profile
    from pstats import SortKey, Stats
        
    with Profile() as profile:
        main()
    
    # get the profile stats
    stats = Stats(profile)
    # remove extraneous paths
    stats.strip_dirs() 
    # sort by cummulative time (or by: SortKey.TIME, SortKey.CALLS)
    stats.sort_stats(SortKey.CUMULATIVE) 
    # filter stats (only top 10)
    # print("-"*100)
    # print(" ")
    # stats.print_stats(10)
    
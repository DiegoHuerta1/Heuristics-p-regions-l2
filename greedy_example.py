from Heuristics.utils import generate_dissimilarity_matrix
import igraph
import os


def main():   
 
    # Define instance
    instance_path = "./Instances_Mexico/07.pkl"
    plots_path = "./Example_plots/"
    os.makedirs(plots_path, exist_ok=True)
    num_regions = 10
    # Reproducibility
    seed = 29
    
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
        "max_generations": 200,
        "tolerance_generations": 328,
        "max_time": 9000,  
        "seed": seed,
        "verbose": True
    }

    # Secuential code (functional version)
    # print("-"*100)
    # print("Functional\n")
    # from Heuristics.brkga_parallel.brkga_functional import Greedy_BRKGA_functional
    # brkga = Greedy_BRKGA_functional(graph, num_regions, diss_matrix, **config)
    # brkga.run()
    # brkga.print_statistics()
    # brkga.plot_evolution(plots_path + "greedy_func_evolution.png") 

    # Parallel (with shared memory)
    print("-"*100)
    print("Parallel (with shared memory)\n")
    from Heuristics.brkga_parallel.brkga_parallel import Greedy_BRKGA_parallel
    brkga = Greedy_BRKGA_parallel(graph, num_regions, diss_matrix, **config,
                                num_workers=5)
    brkga.run()
    brkga.print_statistics()
    brkga.plot_evolution(plots_path + "greedy_parallel_evolution.png") 




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
    print("-"*100)
    print(" ")
    stats.print_stats(10)
    
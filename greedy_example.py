from Heuristics.utils import generate_dissimilarity_matrix
import igraph
import os

from Heuristics.brkga_parallel.Greedy.greedy_brkga_parallel_n import Greedy_BRKGA_parallel_test_n

def main():   
 
    # Define instance
    instance_path = "./Instances_Mexico/07.pkl"
    plots_path = "./Example_plots/"
    os.makedirs(plots_path, exist_ok=True)
    num_regions = 10
    # Reproducibility
    seed = 1
    
    # Read instance
    with open(instance_path, "rb") as f:
        graph = igraph.Graph.Read_Pickle(f)
    print(f"Instance with {graph.vcount()} nodes")
    print(f"Regionalization in {num_regions} regions\n")
    
    # Compute dissimilarity matrix
    diss_matrix = generate_dissimilarity_matrix(graph)
    
    # BRKGA parameters
    config = {
        "population_size": 200,
        "elite_fraction": 0.2,
        "mutant_fraction": 0.2,
        "crossover_rate": 0.7,
        "max_generations": 200,
        "tolerance_generations": 100,
        "max_time": 9000,  
        "seed": seed,
        "rank" : 1,
        "verbose": False,
        "num_workers": 4,
    }

    # -----------------------------------------------------------------------
    brkga = Greedy_BRKGA_parallel_test_n(graph, num_regions, diss_matrix, **config)
    brkga.run()
    brkga.print_statistics()
    brkga.plot_evolution(plots_path + "greedy_brkga_p_numba_evolution.png") 
    brkga.compare_null_weights()
    brkga.ls_best_solution(graph)

    from Heuristics.brkga_core.greedy_brkga import Greedy_BRKGA
    brkga = Greedy_BRKGA(graph, num_regions, diss_matrix, **config)
    brkga.run()
    brkga.print_statistics()
    brkga.plot_evolution(plots_path + "greedy_brkga_p_numba_evolution.png") 


if __name__ == "__main__":
    main()
    
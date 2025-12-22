from Heuristics.utils import generate_dissimilarity_matrix
from Heuristics import MST_BRKGA, ST_BRKGA, Greedy_BRKGA
import igraph
import os


def main():   
 
    # Define instance
    instance_path = "./Instances_Mexico/07.pkl"
    plots_path = "./Example_plots/"
    os.makedirs(plots_path, exist_ok=True)
    num_regions = 10
    
    # Read instance
    with open(instance_path, "rb") as f:
        graph = igraph.Graph.Read_Pickle(f)
    print(f"Instance with {graph.vcount()} nodes")
    print(f"Regionalization in {num_regions} regions\n")
    
    # Compute dissimilarity matrix
    diss_matrix = generate_dissimilarity_matrix(graph)
    
    # BRKGA parameters
    config = {
        "population_size": 1.0,
        "elite_fraction": 0.2,
        "mutant_fraction": 0.2,
        "crossover_rate": 0.7,
        "max_generations": 2000,
        "tolerance_generations": 200,
        "max_time": 9000,  
        "parallel": True,
        "num_workers": 4,
        "seed": 1,
        "verbose": 1
    }
    

    # Apply a MST BRKGA  ------------------------------------------------
    print("-"*100)
    print("MST BRKGA \n")
    brkga = MST_BRKGA(graph, num_regions, diss_matrix, **config)
    brkga.run()
    brkga.print_statistics()
    brkga.plot_evolution(plots_path + "mst_brkga_evolution.png")  


    # Apply a ST BRKGA  ------------------------------------------------
    print("-"*100)
    print("ST BRKGA\n")
    brkga = ST_BRKGA(graph, num_regions, diss_matrix, **config)
    brkga.run()
    brkga.print_statistics()
    brkga.plot_evolution(plots_path + "st_brkga_evolution.png")


    # Apply a Greedy BRKGA  ------------------------------------------------
    print("-"*100)
    print("Greedy BRKGA\n")
    brkga = Greedy_BRKGA(graph, num_regions, diss_matrix, rank = 1, **config)
    brkga.run()
    brkga.print_statistics()
    brkga.plot_evolution(plots_path + "greedy_brkga_evolution.png")


if __name__ == "__main__":
    from cProfile import Profile
    from pstats import SortKey, Stats  
    with Profile() as profile:
        main()
    
    # # get the profile stats
    # stats = Stats(profile)
    # # remove extraneous paths
    # stats.strip_dirs() 
    # # sort by cummulative time (or by: SortKey.TIME, SortKey.CALLS)
    # stats.sort_stats(SortKey.CUMULATIVE) 
    # # filter stats (only top 10)
    # stats.print_stats(10)
    

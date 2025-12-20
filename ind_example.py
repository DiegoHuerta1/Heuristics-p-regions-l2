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
        "population_size": 200,
        "elite_fraction": 0.2,
        "mutant_fraction": 0.2,
        "crossover_rate": 0.7,
        "max_generations": 100,
        "tolerance_generations": 328,
        "max_time": 9000,  
        "seed": seed,
    }
    

    # Apply a MST BRKGA  ------------------------------------------------
    print("-"*100, flush= True)
    print("MST BRKGA \n")

    print("Old")
    from Heuristics.brkga_core_old.specific_brkga import MST_BRKGA as MST_BRKGA_old
    brkga = MST_BRKGA_old(graph, num_regions, diss_matrix, **config)
    brkga.run()
    brkga.print_statistics()
    brkga.plot_evolution(plots_path + "mst_brkga_evolution.png")   

    print("New")
    from Heuristics import MST_BRKGA
    brkga = MST_BRKGA(graph, num_regions, diss_matrix, parallel = True, **config)
    brkga.run()
    brkga.print_statistics()
    brkga.plot_evolution(plots_path + "mst_brkga_evolution.png")  



    # Apply a ST BRKGA  ------------------------------------------------
    print("-"*100, flush= True)
    print("ST BRKGA\n")

    print("Old")
    from Heuristics.brkga_core_old.specific_brkga import ST_BRKGA as ST_BRKGA_old
    brkga = ST_BRKGA_old(graph, num_regions, diss_matrix,  **config)
    brkga.run()
    brkga.print_statistics()
    brkga.plot_evolution(plots_path + "st_brkga_evolution.png")

    print("New")
    from Heuristics import ST_BRKGA
    brkga = ST_BRKGA(graph, num_regions, diss_matrix, parallel = True, **config)
    brkga.run()
    brkga.print_statistics()
    brkga.plot_evolution(plots_path + "st_brkga_evolution.png")


    
    # Apply a Greedy BRKGA  ------------------------------------------------
    print("-"*100, flush= True)
    print("Greedy BRKGA\n")

    print("Old")
    from Heuristics.brkga_core_old.specific_brkga import Greedy_BRKGA as Greedy_BRKGA_old
    brkga = Greedy_BRKGA_old(graph, num_regions, diss_matrix, **config)
    brkga.run()
    brkga.print_statistics()
    brkga.plot_evolution(plots_path + "greedy_brkga_evolution.png")

    print("New")
    from Heuristics import Greedy_BRKGA
    brkga = Greedy_BRKGA(graph, num_regions, diss_matrix, parallel = True, **config)
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
    

import numpy as np
import pandas as pd
import igraph
import os
from pprint import pprint

from Heuristics.utils import generate_dissimilarity_matrix
from Heuristics.brkga_parallel.Greedy.greedy_brkga_parallel_n import Greedy_BRKGA_parallel_test_n


def get_existing_results(path: str) -> list[dict]:
    try:
        df = pd.read_csv(path)
        return df.to_dict(orient='records')
    except:
        return []


def test_config(config: dict, instance: str, num_regions: int):
    """ 
    Apply the greedy brkga with specific parameters
    """
    print("Evaluating the parameters")
    pprint(config)
    
    # Paths
    instance_path = f"./Instances_Mexico/{instance}.pkl"
    output_folder = "./Instances_Mexico/Greedy_Results/"
    plots_path = output_folder + "Plots/"
    df_path = output_folder + "results.csv"
    os.makedirs(plots_path, exist_ok=True)
    # Read instance
    with open(instance_path, "rb") as f:
        graph = igraph.Graph.Read_Pickle(f)
    diss_matrix = generate_dissimilarity_matrix(graph)

    # Get the existing results
    total_results: list[dict] = get_existing_results(df_path)
    id_execution = len(total_results)
    print(f"\nFound {id_execution} previous results\n")

    # Initialize results
    results: dict = config.copy()
    results["Instance"] = instance
    results["N"] = graph.vcount()
    results["K"] = num_regions
    
    # Run
    brkga = Greedy_BRKGA_parallel_test_n(graph, num_regions, diss_matrix, **config)
    brkga.run()
    brkga.print_statistics()
    brkga.plot_evolution(plots_path + f"evolution_{id_execution}.png") 

    # Store results
    results["f"] = brkga.evolution_stats["best_fitness"]
    results["time"] = brkga.evolution_stats["time"]
    results["generations"] = brkga.evolution_stats["population_stats"].shape[0] - 1

    # LS
    ls_results = brkga.ls_best_solution(graph)
    results["time_ls"] = ls_results[0]
    results["iter_ls"] = ls_results[1]
    results["f_LS"] = ls_results[2]

    # Null weights
    results["null_weights"] = brkga.compare_null_weights()

    # Save updated results
    results["ID"] = id_execution
    total_results.append(results)
    df = pd.DataFrame(total_results)
    df.to_csv(df_path, index=False)
    print("\nResults updated!\n")

    
def main():

    config = {
        "population_size": 200,
        "elite_fraction": 0.2,
        "mutant_fraction": 0.2,
        "crossover_rate": 0.7,
        "max_generations": 200,
        "tolerance_generations": 100,
        "max_time": 9000,  
        "seed": 0,
        "rank" : 10,
        "verbose": False,
        "num_workers": 4,
    }
    num_regions = 10
    instance = "07"
    test_config(config, instance, num_regions)


if __name__ == "__main__":
    main()








































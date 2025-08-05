import time
from ..utils import l2_objective_function_diss_matrix, labels_to_P


brkga_heuristics_list = [
    "mst_brkga",
    "st_brkga",
    "greedy_brkga",
]
pygeoda_heuristics_list = [
    "skater",
    "redcap",
    "schc",
    "azp_greedy",
    "azp_sa",
    "azp_tabu"
]
all_heuristics_list = brkga_heuristics_list + pygeoda_heuristics_list
                  

def run_brkga_heuristic(brkga_class, name, graph, num_regions, diss_matrix,
                        brkga_args, dict_results, dict_partitions):
    """
    Run a BRKGA-based heuristic and store metrics and solution in dictionaries
    """
    # Execute
    model = brkga_class(graph, num_regions, diss_matrix, **brkga_args)
    model.run()
    # Metrics
    stats = model.evolution_stats
    dict_results[f"{name}__f"] = stats['best_fitness']
    dict_results[f"{name}__time"] = stats['time']
    dict_results[f"{name}__last_gen"] = stats['population_stats'].index.max()
    # Partition
    dict_partitions[name] = stats["best_solution"]


def run_pygeoda_heuristic(pygeodad_func, name, num_regions, w, data, diss_matrix,
                         pygeoda_args, dict_results, dict_partitions):
    """
    Run a PyGeoda-based heuristic and store metrics and solution in dictionaries
    """
    # Execute
    start = time.time()
    results = pygeodad_func(num_regions, w, data, **pygeoda_args)
    elapsed_time = time.time() - start
    # Metrics
    P = labels_to_P(results["Clusters"], num_regions)
    dict_results[f"{name}__f"] = l2_objective_function_diss_matrix(P, diss_matrix)
    dict_results[f"{name}__time"] = elapsed_time
    # Partition
    dict_partitions[name] = P


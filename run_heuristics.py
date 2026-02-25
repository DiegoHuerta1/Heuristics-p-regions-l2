import matplotlib
import argparse
matplotlib.use("Agg")
import os
os.environ["PYTHONWARNINGS"] = "ignore:resource_tracker:UserWarning"
from Heuristics import Batch_Execution


# ================================================
# PARAMETERS
# ================================================

data_folder = "Instances_Mexico/"
output_folder = "Results_Mexico_1/"  
heuristics = [] # all heuristics
repetitions = 10

brkga_config_1 = {
    "population_size": 1.0,            # equal to the number of genes
    "elite_fraction": 0.2,
    "mutant_fraction": 0.2,
    "crossover_rate": 0.7,
    "max_generations": 1000000,
    "tolerance_generations": 500,
    "max_time": 21600,                 # 6 hours 
    "parallel_brkga": "Auto",          # parallel if num_nodes >= 100
    "num_workers": 10,
    "rank": 1,
    "verbose": 0,
}
brkga_config_2 = {
    "population_size": 500,            
    "elite_fraction": 0.2,
    "mutant_fraction": 0.2,
    "crossover_rate": 0.7,
    "max_generations": 1000000,
    "tolerance_generations": 1000,
    "shaking_tol_generations": 200, 
    "shaking_parameter": 0.25,
    "max_time": 21600,                 # 6 hours 
    "parallel_brkga": "Auto",          # parallel if num_nodes >= 100
    "num_workers": 10,
    "rank": 1,
    "verbose": 0,
}


pygeoda_config = {
    "redcap__method": "fullorder-averagelinkage",
    "schc__linkage_method": "complete",
    "azp_tabu__tabu_length":  10,
}

def get_number_of_regions(n: int) -> list[int]:
    if n <= 10:
        return [3]
    elif n<=20:
        return [5]
    else:
        return [10]


def main():

    # ================================================
    # METHOD EXECUTION
    # ================================================

    model = Batch_Execution(
        brkga_config = brkga_config_1,
        pygeoda_config = pygeoda_config,
        get_k_func = get_number_of_regions,
        data_folder = data_folder,
        output_folder = output_folder,
        save_evolution_plots = True,
        heuristics = heuristics,
        repetitions = repetitions
    )
    model.print_initial_information()

    # Time argument --max_time
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--max_time",
        type = int,
        default = 3600,
        help="Maximum time"
    )
    args = parser.parse_args()
    max_time = args.max_time

    # Run
    if max_time > 0:
        model.run(max_time = max_time)  


    # ================================================
    # ANALYZE RESULTS
    # ================================================

    heuristics_to_analyze = [
    # "mst_brkga",
    "mst_brkga_ls",
    # "msf_brkga",
    "msf_brkga_ls",
    # "st_brkga",
    "st_brkga_ls",
    # "greedy_brkga",
    "greedy_brkga_ls",
    "skater",
    "redcap",
    "schc",
    "azp_greedy",
    "azp_sa",
    "azp_tabu"
    ]
    model.analyze_results(heuristics_to_analyze)


if __name__ == "__main__":
    main()
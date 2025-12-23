import matplotlib
matplotlib.use("Agg")
from Heuristics import Batch_Execution


# ================================================
# PARAMETERS
# ================================================

data_folder = "Instances_Mexico/"
output_folder = "Results_Mexico/"
heuristics = [] 
repetitions = 20

brkga_config = {
    "population_size": 1.0,
    "elite_fraction": 0.2,
    "mutant_fraction": 0.2,
    "crossover_rate": 0.7,
    "max_generations": 20000,
    "tolerance_generations": 200,
    "max_time": 18000,  
    "parallel": True,
    "num_workers": 4,
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
        brkga_config = brkga_config,
        pygeoda_config = pygeoda_config,
        get_k_func = get_number_of_regions,
        data_folder = data_folder,
        output_folder = output_folder,
        save_evolution_plots = True,
        heuristics = heuristics,
        repetitions = repetitions
    )
    model.print_initial_information()
    model.run(max_time = 10000)  


    # ================================================
    # ANALYZE RESULTS
    # ================================================

    heuristics_to_analyze = [
    "mst_brkga",
    "st_brkga",
    "greedy_brkga",
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
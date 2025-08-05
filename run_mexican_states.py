from Heuristics import Batch_Execution
import igraph


# Parameters for the methods
brkga_config = {
    "population_size": 500,
    "elite_fraction": 0.2,
    "mutant_fraction": 0.2,
    "crossover_rate": 0.7,
    "max_generations": 1000,
    "tolerance_generations": 100,
    "max_time": 600,  
    "seed": 10
}
pygeoda_config = {
    "redcap__method": "fullorder-averagelinkage",
    "schc__linkage_method": "complete",
    "azp_tabu__tabu_length":  10,
    "seed": 10,
}

# Numbers of regions L
def get_number_of_regions(n: int) -> list[int]:
    if n < 20:
        return [k for k in [3, 5, 7] if k<n]
    elif n < 60:
        return [k for k in [5, 7, 10] if k<n]
    else:
        return [k for k in [10, 15, 20] if k<n]

# Final settings
data_folder = "./Mexican States Sample Data/"
output_folder = "./Mexican States Results/"
heuristics = []
repetitions = 2

# Execution
model = Batch_Execution(brkga_config, pygeoda_config, get_number_of_regions,
                        data_folder, output_folder, heuristics, repetitions)
model.print_initial_information()
model.run()
model.print_final_information()
print(model.results_df.iloc[:, :10].head())







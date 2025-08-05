from Heuristics import Batch_Execution


# ================================================
# 1. PARAMETERS
# ================================================

# BRKGA parameters
brkga_config = {
    "population_size": 500,
    "elite_fraction": 0.2,
    "mutant_fraction": 0.2,
    "crossover_rate": 0.7,
    "max_generations": 1000,
    "tolerance_generations": 100,
    "max_time": 3600,  
    "seed": 1
}

# PyGeoda configuration
pygeoda_config = {
    "redcap__method": "fullorder-averagelinkage",
    "schc__linkage_method": "complete",
    "azp_tabu__tabu_length":  10,
    "seed": 1,
}

# This function defines how many regions (K values) to try based on the graph size.
def get_number_of_regions(n: int) -> list[int]:
    if n < 20:
        return [k for k in [3, 5, 7] if k<n]
    elif n < 60:
        return [k for k in [5, 7, 10] if k<n]
    else:
        return [k for k in [10, 15, 20] if k<n]


# The data folder should point to the location of your graph instances, all `.pkl` files will be processed.
data_folder = "./Mexican States Sample Data/"

# The output folder defines where to save the analysis.
output_folder = "./Mexican-States-Results/"

# List of heuristics e.g. ["mst_brkga", "pygeoda__azp_sa"] 
# Leave empty to run all available.
heuristics = []  

# Number of times to run each heuristic per instance and per region count.
repetitions = 2


# ================================================
# 2. METHOD EXECUTION
# ================================================

# Initialize the batch execution model
model = Batch_Execution(
    brkga_config = brkga_config,
    pygeoda_config = pygeoda_config,
    get_k_func = get_number_of_regions,
    data_folder = data_folder,
    output_folder = output_folder,
    heuristics = heuristics,
    repetitions = repetitions
)

# Print initial setup summary
model.print_initial_information()

# Run all experiments
# model.run()


# ================================================
# 3. ANALYZE RESULTS
# ================================================


# Print performance summary
model.print_final_information()


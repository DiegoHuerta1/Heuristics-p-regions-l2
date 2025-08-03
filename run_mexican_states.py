from Heuristics import run_all_on_graph
from Heuristics.utils import generate_dissimilarity_matrix
import igraph


# read a graph
id_state = "18"
path_graphs = "./Mexican States Data/Graph Instances/"
with open(path_graphs + f"{id_state}.pkl", "rb") as f:
    graph = igraph.Graph.Read_Pickle(f)


# --------------------

# Parameterss

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

num_regions = 3

diss_matrix = generate_dissimilarity_matrix(graph)

heuristics = []

# --------------------

metrics, partitions = run_all_on_graph(graph = graph, num_regions = num_regions,
                                       brkga_config = brkga_config, pygeoda_config = pygeoda_config,
                                       diss_matrix = diss_matrix, heuristics = heuristics)

from pprint import pprint

print("-"*50)
pprint(metrics)
print("-"*50)


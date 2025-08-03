import igraph
import numpy as np
import geopandas as gpd
from typing import cast
import pygeoda

from ..utils import generate_dissimilarity_matrix, igraph_to_gdf
from ..brkga_core.specific_brkga import MST_BRKGA, ST_BRKGA, Greedy_BRKGA

from .utils import all_heuristics_list, run_brkga_heuristic, run_pygeoda_heuristic


def run_all_on_graph(graph: igraph.Graph, num_regions: int,
                     brkga_config: dict, pygeoda_config: dict,
                     diss_matrix: None | np.ndarray = None,
                     heuristics: list[str] = ["mst_brkga", "st_brkga", "greedy_brkga"]) -> tuple[dict, dict]:
    """
    Run all heuristics on a graph instance

    Args:
        graph (igraph.Graph): Graph instance
        num_regions (int): Number of regions for regionalization
        brkga_config (dict): Parameters for BRKGA methods
        pygeoda_config (dict): Parameters for Pygeoda methods
        diss_matrix (None | np.ndarray, optional): Dissimilarity matrix. Defaults to None.
        heuristics (list[str], optional): List of methods to run. Empty list implies all methods.
                                        Options: ["mst_brkga",
                                                  "st_brkga",
                                                  "greedy_brkga",
                                                  "skater",
                                                  "redcap",
                                                  "schc",
                                                  "azp_greedy",
                                                  "azp_sa",
                                                  "azp_tabu"]
                                        Defaults to ["mst_brkga", "st_brkga", "greedy_brkga"].

    Returns:
        tuple[dict, dict]: Two dictionaries.
            The first one with result metrics for each method
            The second one with the partitions obtained by each method 
    """

    # # Use all heuristics if none specified
    if len(heuristics) == 0:
        heuristics = all_heuristics_list

    # Compute dissimilarity matrix if not present
    if diss_matrix is None:
        diss_matrix = generate_dissimilarity_matrix(graph)

    # Start results dictionary with general information
    dict_results: dict = {
        "N": graph.vcount(),
        "K": num_regions,
    }
    # Start empty partition dictionary
    dict_partitions: dict = {}

    # ----- Run BRKGA methods ----------------------------------

    brkga_methods = {
        "mst_brkga": MST_BRKGA,
        "st_brkga": ST_BRKGA,
        "greedy_brkga": Greedy_BRKGA,
    }
    # Run all methods in heuristics list
    for name, brkga_cls in brkga_methods.items():
        if name in heuristics:
            run_brkga_heuristic(brkga_cls, name, graph, num_regions, diss_matrix,
                                brkga_config, dict_results, dict_partitions)


    # ----- Prepare data for PyGeoda methods ---------------------

    # Transform graph into data and 2
    created_gdf: gpd.GeoDataFrame = igraph_to_gdf(graph)
    gda: pygeoda.gda.geodaGpd = cast(pygeoda.gda.geodaGpd, pygeoda.open(created_gdf))
    w: pygeoda.Weight = pygeoda.queen_weights(gda)
    data = gda[[field for field in gda.field_names if "x_" in field]]

    # Define available PyGeoda methods and arguments
    pygeoda_methods: dict[str, tuple] = {
        "skater": (pygeoda.skater, {}),
        "redcap": (pygeoda.redcap, {"method": pygeoda_config["redcap__method"]}),
        "schc": (pygeoda.schc, {"linkage_method": pygeoda_config["schc__linkage_method"]}),
        "azp_greedy": (pygeoda.azp_greedy, {}),
        "azp_sa": (pygeoda.azp_sa, {}),
        "azp_tabu": (pygeoda.azp_tabu, {"tabu_length": pygeoda_config["azp_tabu__tabu_length"]}),
    }

    # ----- Run PyGeoda heuristics ---------------------------------

    # Run all methods in heuristics list
    for name, (method_func, extra_args) in pygeoda_methods.items():
        if name in heuristics:
            args = {
                "distance_method": "euclidean",
                "scale_method": "raw",
                "random_seed": pygeoda_config["seed"],
                **extra_args
            }
            run_pygeoda_heuristic(method_func, name, num_regions, w, data, diss_matrix,
                                 args, dict_results, dict_partitions)

    return dict_results, dict_partitions




import igraph
import numpy as np
import geopandas as gpd
from typing import cast
import pygeoda
import time
from pathlib import Path
import json

from ..utils import generate_dissimilarity_matrix, igraph_to_gdf
from ..brkga_core.mst_brkga import MST_BRKGA
from ..brkga_core.st_brkga import ST_BRKGA
from ..brkga_core.greedy_brkga import Greedy_BRKGA
from .utils import all_heuristics_list, run_brkga_heuristic, run_pygeoda_heuristic


def run_all_on_graph(graph: igraph.Graph, num_regions: int,
                     brkga_config: dict, pygeoda_config: dict,
                     diss_matrix: None | np.ndarray = None,
                     heuristics: list[str] = ["mst_brkga", "st_brkga", "greedy_brkga"],
                     evolution_folder: str | None = None,
                     max_time: float = float('inf'),
                     partial_path: Path = Path(".")) -> tuple[dict, dict, bool]:
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
        evolution_folder (str | None, optional): Folder to save evolution plots for BRKGA methods.
                                                Defaults to None.
        max_time (float, optional): Maximum time to run all heuristics. Defaults to infinity.
        partial_path (Path, optional): Path to save partial results. Defaults to Path(".").

    Returns:
        tuple[dict, dict, bool]: Two dictionaries and a bool.
            The first one with result metrics for each method
            The second one with the partitions obtained by each method 
            The bool indicates if the results are complete (i.e., all methods were run)
    """
    start_time = time.time()

    # Use all heuristics if none specified
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

    # Check por partial results
    if partial_path.exists():
        with open(partial_path, "r") as f:
            partial_results = json.load(f)
        dict_results.update(partial_results["metrics"])
        dict_partitions.update(partial_results["partitions"])
        print(f"\tLoaded partial results from {partial_path}", flush=True)
        # Update heuristics to run
        heuristics = [h for h in heuristics if f"{h}__f" not in dict_results]


    # ----- Run BRKGA methods ----------------------------------

    brkga_methods = {
        "mst_brkga": MST_BRKGA,
        "st_brkga": ST_BRKGA,
        "greedy_brkga": Greedy_BRKGA,
    }
    # Run all methods in heuristics list
    for name, brkga_cls in brkga_methods.items():
        if name in heuristics:
            # Path to save evolution plot
            evolution_path: str | None = None
            if evolution_folder is not None:
                evolution_path = evolution_folder + f"{name}__{brkga_config['seed']}.png"
            # Execute
            run_brkga_heuristic(brkga_cls, name, graph, num_regions, diss_matrix,
                                brkga_config, dict_results, dict_partitions,
                                evolution_path)
            
            # Save partial results
            partial_results = {"metrics": dict_results, "partitions": dict_partitions}
            with open(partial_path, "w") as f:
                json.dump(partial_results, f)

            # Check max time
            elapsed_time = time.time() - start_time
            if elapsed_time >= max_time:
                print(f"\tMaximum time reached. Incomplete instance after {name}", flush=True)
                return dict_results, dict_partitions, False


    # ----- Prepare data for PyGeoda methods ---------------------

    # Transform graph into data and w
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
            
            # Save partial results
            partial_results = {"metrics": dict_results, "partitions": dict_partitions}
            with open(partial_path, "w") as f:
                json.dump(partial_results, f)

            # Check max time
            elapsed_time = time.time() - start_time
            if elapsed_time >= max_time:
                print(f"\tMaximum time reached. Incomplete instance after {name}", flush=True)
                return dict_results, dict_partitions, False
            
    # All methods completed
    if partial_path.exists():
        partial_path.unlink()
    return dict_results, dict_partitions, True




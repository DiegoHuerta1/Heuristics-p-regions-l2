import time
from ..utils import l2_objective_function_diss_matrix, labels_to_P
import pandas as pd

brkga_heuristics_list = [
    "mst_brkga",
    "msf_brkga",
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
                        brkga_args, dict_results, dict_partitions, evolution_path):
    """
    Run a BRKGA-based heuristic and store metrics and solution in dictionaries
    """
    # Execute
    model = brkga_class(graph, num_regions, diss_matrix, **brkga_args)
    model.run()
    # Save evolution plot if required
    if evolution_path is not None:
        model.plot_evolution(evolution_path)
    # Metrics
    stats = model.evolution_stats
    dict_results[f"{name}__f"] = stats['best_fitness']
    dict_results[f"{name}__time"] = stats['time']
    dict_results[f"{name}__last_gen"] = stats['population_stats'].index.max()
    # Chromosome length
    dict_results[f"{name}__c_len"] = model.n
    # Local Search improvement
    model.ls_improvement(graph)
    ls_stats = model.ls_stats
    dict_results[f"{name}__f_ls"] = ls_stats['f_P']
    dict_results[f"{name}__time_ls"] = ls_stats['time']
    dict_results[f"{name}__iterations_ls"] = len(ls_stats['historial_f']) - 1
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



def get_ranks_df(heuristics: list[str], df_results: pd.DataFrame) -> pd.DataFrame:
    """ 
    Transorm a df of results to ranks
    """
    if df_results.shape[0] > 0:
        # Filter results of desired heuristics and compute ranks
        columns_f = [f"{h}__f" for h in heuristics]
        return df_results[columns_f].round(8).rank(axis = 1)
    else:
        return pd.DataFrame()



def add_relative_gap_columns(df: pd.DataFrame, heuristics: list[str]) -> pd.DataFrame:
    """ 
    Add a column with the relative gap for each heuristic.
    The relative gap is computed as:
        (f_heuristic - f_best) / f_best
    The f_best is calculated among all heuristics and all repetitions for each [Name, K].
    """

    # Helpful 
    df = df.copy()
    df["Name"] = df["ID"].apply(lambda x: x.split("__")[0])
    f_cols = [f"{h}__f" for h in heuristics]
    df["row_min_f"] = df[f_cols].apply(min, axis = 1)

    # Compute the best for each [Name, K]
    df_best_f = df.groupby(["Name", "K"])["row_min_f"].min()
    # Insert in the original df
    df = df.merge(
        df_best_f.rename("best_f_overall"),
        left_on = ["Name", "K"],
        right_index = True
    )

    # Compute relative gaps and add columns
    for h in heuristics:
        f_col = f"{h}__f"
        rel_gap_col = f"{h}__rel_gap"
        df[rel_gap_col] = 100 * (df[f_col] - df["best_f_overall"]) / df["best_f_overall"]

    # Drop helper columns
    df = df.drop(columns = ["Name", "row_min_f", "best_f_overall"])
    return df



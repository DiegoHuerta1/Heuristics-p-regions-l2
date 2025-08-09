import time
from ..utils import l2_objective_function_diss_matrix, labels_to_P
import numpy as np
import seaborn as sns
from scipy.stats import friedmanchisquare
import scikit_posthocs as sp
import pandas as pd
import matplotlib.colors as mcolors


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



def analyze_friedman(df: pd.DataFrame, primary_factors: list[str], metric: str, 
                     ax_sign_plot, ax_cd_diagram, 
                     verbose=False):
    """
    Perform a Friedman test and post-hoc analysis for a block design experiment.
    
    Parameters
    ----------
    df : pandas.DataFrame
        Input data containing blocking factors as rows (e.g., problem instances),
        and measurements of each primary factor (e.g., heuristic) for the given metric
        as columns. Each primary factor column should follow the naming pattern:
        '{primary_factor}__{metric}'.
        
    primary_factors : list of str
        Names of the primary factors (without metric suffix).
        
    metric : str
        Name of the metric being evaluated.
        
    ax_sign_plot : matplotlib.axes.Axes
        Axis on which to draw the significance plot.
        
    ax_cd_diagram : matplotlib.axes.Axes
        Axis on which to draw the critical difference diagram.
        
    verbose : bool, default=False
        If True, prints detailed statistics.
    
    Returns
    -------
    float
        p-value from the Friedman test.
    """
    
    # Select relevant columns for the given metric
    target_columns = [f"{p}__{metric}" for p in primary_factors]
    df_metric = df[target_columns].round(5)  # copy subset and round
    
    # Rename columns to only primary factor names (remove metric suffix)
    rename_map = {f"{p}__{metric}": p for p in primary_factors}
    df_renamed = df_metric.rename(columns=rename_map)
    
    # Compute ranks for each row (block)
    rank_df = df_renamed.rank(axis=1)

    # Perform Friedman test
    stat, p_value = friedmanchisquare(*df_renamed.values.T)
    
    if p_value < 0.05:
        # Post-hoc Nemenyi test
        test_results = sp.posthoc_nemenyi_friedman(df_renamed) + np.finfo(float).eps
        
        # Significance plot
        sp.sign_plot(test_results, ax=ax_sign_plot,
                     cmap=["white", "#FFA7A7", "#1B5E20", "#4CAF50", "#A8E6A3"])
        
        # Choose color palette
        if len(primary_factors) > 10:
            palette = sns.color_palette("husl", len(primary_factors))
        else:
            palette = sns.color_palette()
        color_mapping: dict[str, str] = {
            f"{p}": mcolors.to_hex(palette[i]) for i, p in enumerate(primary_factors)
        }
        
        # Critical difference diagram
        sp.critical_difference_diagram(rank_df.mean(axis=0),
                                       test_results,
                                       color_palette=color_mapping,
                                       ax=ax_cd_diagram)
    
    if verbose:
        print("")
        print("-"*100)
        print(f"Friedman test for differences in {metric}")
        print(f"Statistic: {np.round(stat, 4)}")
        print(f"P-value: {p_value}")
        if p_value < 0.05:
            print("\nPost-hoc results (Nemenyi test):")
            print(test_results.round(2))
        print("-"*100)
    
    return p_value
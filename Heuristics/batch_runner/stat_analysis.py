import numpy as np
import seaborn as sns
from scipy.stats import friedmanchisquare
import scikit_posthocs as sp
import pandas as pd
import matplotlib.colors as mcolors
import numpy as np
import matplotlib.pyplot as plt

from .utils import get_ranks_df


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
            print(test_results.round(2)) # type: ignore
        print("-"*100)
    
    return p_value


def friedman_for_heuristics(heuristics: list[str], df_analyze: pd.DataFrame,
                            plot_folder: str):
    """  
    Perform Friedman test to analyze differences in performance and execution time
    of heuristics
    """

    # Analyze differences in final result (f)
    fig, axes = plt.subplot_mosaic(
        [["Big"],
        ["Big"],
        ["Small"]],
        figsize = (10, 12), dpi = 300
    )
    analyze_friedman(df_analyze, heuristics, "f",
                        ax_sign_plot = axes["Big"], # type: ignore
                        ax_cd_diagram = axes["Small"],  # type: ignore
                        verbose=True)
    fig.suptitle("Differences in final f", fontsize=26, fontweight='bold')
    plt.savefig(plot_folder + 'Differences_f.png', bbox_inches='tight')
    plt.close()

    # Analyze differences in time
    fig, axes = plt.subplot_mosaic(
        [["Big"],
        ["Big"],
        ["Small"]],
        figsize = (10, 12), dpi = 300
    )
    analyze_friedman(df_analyze, heuristics, "time",
                        ax_sign_plot = axes["Big"], # type: ignore
                        ax_cd_diagram = axes["Small"], # type: ignore
                    verbose=True)
    fig.suptitle("Differences in final time", fontsize=26, fontweight='bold')
    plt.savefig(plot_folder + 'Differences_time.png', bbox_inches='tight')
    plt.close()






from typing import Callable
import igraph
from pathlib import Path
import os
from tqdm import tqdm
import pandas as pd
import json
import time

from .utils import all_heuristics_list, get_ranks_df, add_relative_gap_columns
from .stat_analysis import friedman_for_heuristics
from ..utils import generate_dissimilarity_matrix, compute_P_names
from .run_all import run_all_on_graph
from ..visualizations.plot_results import ResultPlotter


class Batch_Execution():
    """ 
    Class for executing heuristics on a folder of instances and analyze results.

    Works for graphs saved as a .pkl file
    """

    def __init__(self, brkga_config: dict, pygeoda_config: dict,
                 get_k_func: Callable[[int], list[int]], 
                 data_folder: str, output_folder: None | str = None,
                 save_evolution_plots: bool = True,
                 heuristics: list[str] = [],
                 repetitions: int = 3 ):
        """
        Class constructor

        Args:
            brkga_config (dict): Shared arguments for brkga heuristics (not graph, num_reg, diss_matrix)
            pygeoda_config (dict): Shared arguments for pygeoda heuristics (not num_regions, w, data)
            get_k_func (Callable[[int], list[int]]): A function that takes the size of an instance and 
                                                     return a list of numbers of regions to perform regionalization
            data_folder (str): Path and name of folder full of graph instances
            output_folder (None | str, optional): Folder to store the results. Defaults to None.
            save_evolution_plots (bool, optional): Whether to save evolution plots for BRKGA methods.
            heuristics (list[str], optional): List of heuristics to apply, empty list indicates all of them. Defaults to [].
            repetitions (int, optional): How many repetitions to perform for eahc (instance, num_regions). Defaults to 3.
        """

        # General atributes
        self.brkga_config: dict = brkga_config
        self.pygeoda_config: dict = pygeoda_config
        self.get_k_func: Callable[[int], list[int]] = get_k_func
        self.repetitions: int = repetitions
        self.heuristics: list[str] = heuristics if heuristics else all_heuristics_list
        # Add ls version for the brkga
        brkga_h = [h for h in self.heuristics if "brkga" in h]
        self.heuristics += [f"{h}_ls" for h in brkga_h]

        # Input data and instances
        self.data_folder: str = data_folder
        self.instances: dict[str, igraph.Graph] = self.get_instances()

        # Results folder and path
        if output_folder is None:
            output_folder = data_folder + "Results/"
        self.output_folder: str = output_folder
        os.makedirs(output_folder, exist_ok=True)
        self.output_df_path: Path = Path(self.output_folder) / "df_results.csv"

        # Results data (list of dicts, df)
        self.results: list[dict] = self.get_partial_results()
        self.results_df: pd.DataFrame = self.get_results_df(self.results)

        # Progress folder (ids) and files
        self.ids_folder: str = output_folder + "Progress/"
        os.makedirs(self.ids_folder, exist_ok=True)
        self.all_ids_path: Path = Path(self.ids_folder) / "all_ids.txt"
        self.completed_ids_path: Path = Path(self.ids_folder) / "completed_ids.txt"

        # Lists of all and completed ids
        self.all_ids: list[str] = self.get_all_ids()
        self.completed_ids: list[str] = self.get_completed_ids()

        # Aditional folders 
        self.plot_folder: str = output_folder + "Plots/"
        self.partitions_folder: str = output_folder + "Partitions/"
        os.makedirs(self.plot_folder, exist_ok=True)
        os.makedirs(self.partitions_folder, exist_ok=True)

        # Save Parameters
        self.parameters_folder: str = output_folder + "Parameters/"
        os.makedirs(self.parameters_folder, exist_ok=True)
        brkga_params_path: Path = Path(self.parameters_folder) / "brkga_config.json"
        pygeoda_params_path: Path = Path(self.parameters_folder) / "pygeoda_config.json"
        with open(brkga_params_path, "w") as json_file:
            json.dump(self.brkga_config, json_file, indent=4)
        with open(pygeoda_params_path, "w") as json_file:
            json.dump(self.pygeoda_config, json_file, indent=4)

        # Save evolution plots option
        self.save_evolution_plots: bool = save_evolution_plots
        self.evolution_folder: str | None = None
        if self.save_evolution_plots:
            self.evolution_folder = output_folder + "Evolution_Plots/"
            os.makedirs(self.evolution_folder, exist_ok=True)


    def print_initial_information(self):
        """ 
        Print general information
        """
        print("-"*50)
        print(f"Batch run considering {len(self.heuristics)} heuristics")
        print(f"Data folder with {len(self.instances)} instances")
        print(f"{self.repetitions} repetitions for each pair (instance, num_regions)\n")
        print(f"Total of {len(self.all_ids)} different executions for each heuristic")
        print("-"*50)

    # ---------------------------
    # Instances and Ids ---------
    # ---------------------------

    def get_instances(self) -> dict[str, igraph.Graph]:
        """ 
        Get all the instances from the data folder.
        Save in dictionary {name_intance: graph}
        """
        instances = {}
        # Iterate on all .pkl files
        pkl_files = Path(self.data_folder).glob("*.pkl")
        for file_path in pkl_files:
            try:
                # Read graph
                graph = igraph.Graph.Read_Pickle(str(file_path))
                name = file_path.stem  
                instances[name] = graph
            except Exception as e:
                print(f"Failed to read {file_path.name}: {e}")
        return instances


    def get_all_ids(self) -> list[str]:
        """ 
        An id is of the form x__y__z where
            x: the name of the instance
            y: the number of regions
            z: the repetition (and seed)

        If an 'all_ids.txt' file exists in the progress folder, read from it.
        Otherwise, compute the IDs and save them to that file.
        """
        # If cached version exists, read and return it
        if self.all_ids_path.exists():
            with open(self.all_ids_path, "r") as f:
                ids = [line.strip() for line in f if line.strip()]
            return ids

        # Otherwise, generate all IDs
        ids = []
        # For each instance
        for instance_name, graph in self.instances.items():
            # Use the size to define the number of regions
            n = graph.vcount()
            list_num_regions = self.get_k_func(n)
            for num_regions in list_num_regions:
                # Repeat several times
                for repetition in range(self.repetitions):
                    # Create an id
                    ids.append(f"{instance_name}__{num_regions}__{repetition}")

        # Save to file
        with open(self.all_ids_path, "w") as f:
            f.write("\n".join(ids))
        return ids


    def get_completed_ids(self) -> list[str]:
        """ 
        Try to restore the ids that have been marked as completed
        """
        # If cached version exists, read and return it
        if self.completed_ids_path.exists():
            with open(self.completed_ids_path, "r") as f:
                completed_ids = [line.strip() for line in f if line.strip()]
            return completed_ids
        # Otherwise, not a single id has been completed
        return []
    
    # ---------------------------
    # Manage results  ----------
    # ---------------------------

    def get_partial_results(self) -> list[dict]:
        """ 
        Try to get results from the output folder
        If not then partial results are empty
        """
        try:
            df_partial_results = pd.read_csv(self.output_df_path)
            return df_partial_results.to_dict(orient="records")
        except:
            return []
        

    def order_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """ 
        Order the df columns.
        First general information, then _f for each method, then time,
        at the end additional information.
        """
        # Identify the first columns
        info_columns: list[str] = ["ID", "N", "K"]
        f_columns: list[str] = [f"{h}__f" for h in self.heuristics]
        time_columns: list[str] = [f"{h}__time" for h in self.heuristics]
        general_columns: list[str] = info_columns + f_columns + time_columns
        # Any other information
        aditional_info: list[str] = [col for col in df.columns if col not in general_columns]
        # Order columns
        col_order: list[str] = info_columns + f_columns + time_columns + aditional_info
        return df[col_order]


    def get_results_df(self, results: list[dict]) -> pd.DataFrame:
        """ 
        Transform results from list of dicts to a pandas dataframe
        """
        if results:
            return self.order_columns(pd.DataFrame(results))
        else:
            return pd.DataFrame()

    def save_results_iteration(self, id_: str, metrics: dict, partitions: dict, graph: igraph.Graph):
        """ 
        After completed the execution of an id
        Save the resutls and mark the id as complete
        """
        # Mark id as complete
        self.completed_ids.append(id_)
        with open(self.completed_ids_path, "w") as f:
            f.write("\n".join(self.completed_ids))

        # Save metric results and update df
        self.results.append(metrics)
        self.results_df = self.get_results_df(self.results)
        self.results_df = add_relative_gap_columns(self.results_df, self.heuristics)
        self.results_df.to_csv(self.output_df_path, index=False)

        # Save partitions with names of nodes, not index
        for method, P in partitions.items():
            P_names = compute_P_names(graph, P)
            partition_path = Path(self.partitions_folder) / f"{method}__{id_}.txt"
            with open(partition_path, "w") as json_file:
                json.dump(P_names, json_file, indent=4)


    # ---------------------------
    # Run executions ------------
    # ---------------------------

    def run(self, max_time: float = float('inf')):
        """ 
        Main function.
        Starts the execution loop of all the methods in all the instances.
        Args:
            max_time (float, optional): Maximum time for the whole execution loop. Defaults to infinity.
        """
        start_time = time.time()

        # Check if any progress has been made before
        self.completed_ids = self.get_completed_ids()
        self.results = self.get_partial_results()

        # Iterate on not completed ids
        for id_ in tqdm(self.all_ids):
            if id_ in self.completed_ids:
                time.sleep(0.005)
                continue

            # Get elements of the execution from id name
            instance_name = id_.split("__")[0]
            num_regions = int(id_.split("__")[1])
            repetition = int(id_.split("__")[2])

            # Get additional elements for this execution
            graph = self.instances[instance_name]
            diss_matrix =  generate_dissimilarity_matrix(graph)
            brkga_config = self.brkga_config.copy()
            brkga_config["seed"] = repetition
            pygeoda_config = self.pygeoda_config.copy()
            pygeoda_config["seed"] = repetition
            # Folder of the evolutions for this instance
            instance_evolution_folder: str | None = None
            if self.evolution_folder is not None:
                instance_evolution_folder = self.evolution_folder + f"{instance_name}__K_{num_regions}/"
                os.makedirs(instance_evolution_folder, exist_ok=True)

            # Execute heuristics
            remaining_time = max_time - (time.time() - start_time)
            metrics, partitions, complete = run_all_on_graph(graph, num_regions,
                                                             brkga_config, pygeoda_config,
                                                             diss_matrix, self.heuristics,
                                                             instance_evolution_folder,
                                                             max_time = remaining_time,
                                                             partial_path = Path(self.ids_folder)/f"partial_{id_}.json")
            metrics["ID"] = id_

            # Check for incomplete results
            if not complete:
                break

            # Mark this execution id as complete
            self.save_results_iteration(id_, metrics, partitions, graph)

            # Check max time to continue
            elapsed_time = time.time() - start_time
            if elapsed_time >= max_time:
                print(f"\tMaximum time reached. Last instance: {id_}")
                break


    # ---------------------------
    # Analyze Results -----------
    # ---------------------------

    def analyze_results(self,  heuristics_: list[str] = []):
        """ 
        Summarize performance, statistical test and visualizations after execution.
    
        """
        # In case there are no results yet
        if self.results_df.empty:
            print("No results to analyze.")
            print("-" * 50)
            return
        
        # Use full heuristics if none specified
        heuristics: list[str] = heuristics_ if len(heuristics_) > 0 else self.heuristics
        # Get list of instances in the results
        instances: list[str] = list(self.results_df["ID"].apply(lambda x: x.split("__")[0]).unique())        

        print("")
        print("-"*100)
        print(f"Analyzing results for {len(heuristics)} heuristics and {len(instances)} instances")
        print("Heuristics:", ", ".join(heuristics))
        print("Instances:", ", ".join(instances))
        print(f"Total of {self.results_df.shape[0]} executions\n")

        # Analize for each instance
        for instance_name in instances:
            df_analyze = self.results_df[self.results_df["ID"].apply(lambda x: x.split("__")[0] == instance_name)]

            # Plot results for this instance
            plot_results = ResultPlotter(heuristics, df_analyze, self.plot_folder, instance_name)
            plot_results.make_all_plots()

            # Print performance report for each K
            K_values = df_analyze["K"].unique()
            for K in K_values:
                df_analyze_K = df_analyze[df_analyze["K"] == K]
                print("\n" + "-"*100)
                print(f"Instance: {instance_name} - K = {K}")
                self._print_heuristics_performance_report(heuristics, df_analyze_K)


        # Agregate analysis over all instances
        print("\n" + "-"*100)
        print("All Instances aggregated")
        self._print_heuristics_performance_report(heuristics, self.results_df)
        plot_results = ResultPlotter(heuristics, self.results_df, self.plot_folder, "All")
        plot_results.make_all_plots()
        if len(heuristics) >= 3:
            friedman_for_heuristics(heuristics, self.results_df, self.plot_folder)


    # helper
    def _print_heuristics_performance_report(self, heuristics: list[str],
                                             df_analyze: pd.DataFrame):
        """ 
        Performance information for a subset of heuristics and instances
        """

        # Compute mean objective value 
        f_means = {
            h: df_analyze[f"{h}__f"].mean()
            for h in heuristics
        }
        # Compute min objective value 
        f_mins = {
            h: df_analyze[f"{h}__f"].min()
            for h in heuristics
        }
        # Compute mean relative gap 
        rel_gap_means = {
            h: df_analyze[f"{h}__rel_gap"].mean()
            for h in heuristics
        }
        # Compute mean rank
        ranks_df: pd.DataFrame = get_ranks_df(heuristics, df_analyze)
        mean_ranks = ranks_df.mean()
        # Win count per heuristic (lowest rank in each row)
        min_ranks = ranks_df.min(axis=1)
        win_counts = (ranks_df.eq(min_ranks, axis=0)).sum()
    
        # Combine all stats into a summary table
        summary = []
        for h in heuristics:
            summary.append({
                "heuristic": h,
                "min_f": f_mins.get(h, float('nan')),
                "mean_f": f_means.get(h, float('nan')),
                "mean_rel_gap": rel_gap_means.get(h, float('nan')),
                "mean_rank": mean_ranks.get(f"{h}__f", float('nan')),
                "wins": win_counts.get(f"{h}__f", 0)
            })

        # Print summary table
        print("-" * 100)
        print("Summary table\n")
        print(f"{'Heuristic':<20} {'Min f':>10} {'Mean f':>10} {'Mean (rel) gap':>15} {'Mean Rank':>12} {'Wins':>8}")
        for row in sorted(summary, key=lambda x: x["mean_rank"]):
            print(f"{row['heuristic']:<20} {row['min_f']:>10.4f} {row['mean_f']:>10.4f}  {row['mean_rel_gap']:>15.4f} {row['mean_rank']:>12.4f} {row['wins']:>8}")
        print("-" * 100)


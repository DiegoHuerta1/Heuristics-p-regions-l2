from typing import Callable
import igraph
from pathlib import Path
import os
from tqdm import tqdm
import pandas as pd
import json
import time


from .utils import all_heuristics_list
from ..utils import generate_dissimilarity_matrix, compute_P_names
from .run_all import run_all_on_graph


class Batch_Execution():
    """ 
    Class for executing heuristics on a folder of instances and analyze results.

    Works for graphs saved as a .pkl file
    """

    def __init__(self, brkga_config: dict, pygeoda_config: dict,
                 get_k_func: Callable[[int], list[int]], 
                 data_folder: str, output_folder: None | str = None,
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
            heuristics (list[str], optional): List of heuristics to apply, empty list indicates all of them. Defaults to [].
            repetitions (int, optional): How many repetitions to perform for eahc (instance, num_regions). Defaults to 3.
        """

        # General atributes
        self.brkga_config = brkga_config
        self.pygeoda_config = pygeoda_config
        self.get_k_func = get_k_func
        self.repetitions = repetitions

        # Data and instances
        self.data_folder = data_folder
        self.instances: dict[str, igraph.Graph] = self.get_instances()

        # Results folder and path
        if output_folder is None:
            output_folder = data_folder + "Results/"
        self.output_folder = output_folder
        os.makedirs(output_folder, exist_ok=True)
        self.output_df_path = Path(self.output_folder) / "df_results.csv"
        self.results: list[dict] = self.get_partial_results()
        if self.results:
            self.results_df: pd.DataFrame = self.order_columns(pd.DataFrame(self.results))
        else:
            self.results_df: pd.DataFrame = pd.DataFrame()

        # Progress folder (ids) and files
        self.ids_folder = output_folder + "Progress/"
        os.makedirs(self.ids_folder, exist_ok=True)
        self.all_ids_path = Path(self.ids_folder) / "all_ids.txt"
        self.completed_ids_path = Path(self.ids_folder) / "completed_ids.txt"
        self.all_ids: list[str] = self.get_all_ids()
        self.completed_ids: list[str] = self.get_completed_ids()

        # Aditional folders 
        self.plot_folder = output_folder + "Plots/"
        self.partitions_folder = output_folder + "Partitions/"
        os.makedirs(self.plot_folder, exist_ok=True)
        os.makedirs(self.partitions_folder, exist_ok=True)

        # Use all heuristics if none specified
        self.heuristics: list[str] = heuristics if heuristics else all_heuristics_list


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
        
    
    def save_results_iteration(self, id_: str, metrics: dict, partitions: dict, graph: igraph.Graph):
        """ 
        After completed the execution of an id
        Save the resutls and mark the id as complete
        """

        # Mark id as complete
        self.completed_ids.append(id_)
        with open(self.completed_ids_path, "w") as f:
            f.write("\n".join(self.completed_ids))

        # Save metric results from this execution id
        self.results.append(metrics)
        self.results_df = self.order_columns(pd.DataFrame(self.results))
        self.results_df.to_csv(self.output_df_path, index=False)

        # Save partitions with names of nodes, not index
        for method, P in partitions.items():
            P_names = compute_P_names(graph, P)
            partition_path = Path(self.partitions_folder) / f"{method}__{id_}.txt"
            with open(partition_path, "w") as json_file:
                json.dump(P_names, json_file, indent=4)


    def order_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """ 
        Order the df columns.
        First general information, then _f for each method, then time,
        at the end additional information.
        """
        # Identify sets of columns
        general_info = ["ID", "N", "K"]
        f_columns = [c for c in df.columns if c.endswith("__f")]
        time_columns = [c for c in df.columns if c.endswith("__time")]
        # Aditional information
        known_cols = set(general_info + f_columns + time_columns)
        aditional_info = [col for col in df.columns if col not in known_cols]
        # Order
        col_order = general_info + f_columns + time_columns + aditional_info
        return df[col_order]
        

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


    def print_final_information(self):
        """ 
        Summarize performance after execution
        """
        print("-"*50)
        print(f"Completed {len(self.completed_ids)} executions for each heuristic.")
        if self.results_df.empty:
            print("No results to summarize.")
            print("-" * 50)
            return
        
        # Collect mean and median for each heuristic
        stats = []
        for heuristic in self.heuristics:
            f_col = f"{heuristic}__f"
            if f_col not in self.results_df.columns:
                continue
            values = self.results_df[f_col]
            stats.append({
                "heuristic": heuristic,
                "mean": values.mean(),
                "median": values.median()
            })

        # Sort and print mean table
        print("\nMean objective value:")
        print(f"{'Heuristic':<20} {'Mean':>10}")
        for row in sorted(stats, key=lambda x: x["mean"]):
            print(f"{row['heuristic']:<20} {row['mean']:>10.4f}")

        print("\nMedian objective value:")
        print(f"{'Heuristic':<20} {'Median':>10}")
        for row in sorted(stats, key=lambda x: x["median"]):
            print(f"{row['heuristic']:<20} {row['median']:>10.4f}")

        print("-"*50)


    def run(self):
        """ 
        Main function.
        Starts the execution loop of all the methods in all the instances.
        """

        # Check if any progress has been mande before
        self.completed_ids = self.get_completed_ids()
        self.results = self.get_partial_results()

        # Iterate on not completed ids
        for id_ in tqdm(self.all_ids):
            if id_ in self.completed_ids:
                time.sleep(0.01)
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

            # Execute heuristics
            metrics, partitions = run_all_on_graph(graph, num_regions,
                                                   brkga_config, pygeoda_config,
                                                   diss_matrix, self.heuristics)
            metrics["ID"] = id_

            # Mark this execution id as complete
            self.save_results_iteration(id_, metrics, partitions, graph)


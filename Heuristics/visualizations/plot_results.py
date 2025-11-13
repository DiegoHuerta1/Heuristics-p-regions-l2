import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os



class ResultPlotter:
    """ 
    Class for plotting results of heuristic experiments (batch runs).
    """

    def __init__(self, heuristics: list[str], df: pd.DataFrame, plot_folder: str):
        """  
        Args:
            df (pd.DataFrame): DataFrame with results of heuristics.
                Format used for BatchRunner results (internally, with relative gap).
                Columns: ID, N, K, ...
                ID = Name__K__Repetition
            heuristics (list[str]): List of heuristic names.
            plot_folder (str): Folder to save plots.
        """
        self.df: pd.DataFrame = df
        self.heuristics: list[str] = heuristics
        self.plot_folder: str = plot_folder
        self.ind_folder = self.plot_folder + "Individual_plots/"
        os.makedirs(self.plot_folder, exist_ok=True)
        os.makedirs(self.ind_folder, exist_ok=True)

        # Name of all instances (graphs)
        self.instance_names = self._get_all_names()

        # Columns for plotting
        self.f_columns = [f"{h}__f" for h in self.heuristics]
        self.relative_gap_columns = [f"{h}__rel_gap" for h in self.heuristics]
        self.time_columns = [f"{h}__time" for h in self.heuristics]

        # Meltet dataframes for plotting

        self.df_melted_f = self._melt_df(self.f_columns, "Objective", "f")
        self.df_melted_time = self._melt_df(self.time_columns, "Time", "time")
        self.df_melted_rel_gap = self._melt_df(self.relative_gap_columns, "Relative Gap", "rel_gap")

    # Helper Functions ----------------------------------

    def _melt_df(self, value_vars: list[str], value_name: str, suffix: str) -> pd.DataFrame:
        """  
        Melt the dataframe (in a particular variable) for plotting.
        """
        df_melted = self.df.melt(
            id_vars=["ID", "N", "K"],
            value_vars = value_vars,
            var_name="Heuristic",
            value_name= value_name,
        )
        df_melted["Heuristic"] = df_melted["Heuristic"].str.replace(f"__{suffix}", "", regex=False)
        return df_melted

    def _get_all_names(self) -> list[str]:
        """ 
        Get the names of all the instances (graphs)
        """
        return list(self.df["ID"].apply(lambda x: x.split("__")[0]).unique())
    

    def filter_by_instance(self, df: pd.DataFrame, instance_name: str) -> pd.DataFrame:
        """ 
        Get the results for a specific instance (graph)
        """
        return df[df["ID"].apply(lambda x: x.split("__")[0] == instance_name)]
    
    # Main visualization function -----------------------------

    def make_all_plots(self):

        # Boxplots of f and time: all executions
        self.make_boxplot(self.df_melted_f, "Objective", "Objective - all instances", self.plot_folder + "f.png")
        self.make_boxplot(self.df_melted_time, "Time", "Time - all instances", self.plot_folder + "time.png")
        # Relative gap
        self.make_barplot_with_text_by_K(self.df_melted_rel_gap, "Relative Gap",
                                        "(avg) Relative Gap - all instances", self.plot_folder + "relative_gap.png")

        # Boxplots of f and time: for each instances
        for name in self.instance_names:
            instance_folder = self.ind_folder + f"{name}/"
            os.makedirs(instance_folder, exist_ok=True)
            # Boxplot of time and f
            self.make_boxplot(self.filter_by_instance(self.df_melted_f, name), "Objective",
                             f"Objective - instance {name}", instance_folder + f"f_{name}.png")
            self.make_boxplot(self.filter_by_instance(self.df_melted_time, name), "Time",
                             f"Time - instance {name}", instance_folder + f"time_{name}.png")
            # Relative gap
            self.make_barplot_with_text_by_K(self.filter_by_instance(self.df_melted_rel_gap, name), "Relative Gap",
                                             f"(avg) Relative Gap - instance {name}", instance_folder + f"rel_gap_{name}.png")


    # Individual plot functions ---------------------------------

    def make_boxplot(self, df_plot: pd.DataFrame, y: str, title: str, filepath: str):
        """ 
        Make a boxplot for the given data.

        Args:
            df_plot (pd.DataFrame): DataFrame with data to plot (melted).
            y (str): Column name for y-axis.
            title (str): Title of the plot.
            filepath (str): Filepath to save the plot.
        """
        plt.figure(figsize=(12, 6))
        sns.boxplot(data = df_plot, hue="Heuristic", y=y)
        plt.title(title)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(filepath)
        plt.close()

    def make_barplot_with_text_by_K(self, df_plot: pd.DataFrame, y: str, title: str, filepath: str):
        """ 
        Make a barplot with text annotations for the given data.
        The bars are grouped by K.

        Args:
            df_plot (pd.DataFrame): DataFrame with data to plot (melted).
            y (str): Column name for y-axis.
            title (str): Title of the plot.
            filepath (str): Filepath to save the plot.
        """
        df_plot = df_plot.copy()
        df_plot["K"] = df_plot["K"].astype(str)

        g = sns.catplot(df_plot, x="Heuristic", y= y, kind="bar", hue = "K", 
                        height=6, aspect=2, errorbar=None)
        # Annotations
        for ax in g.axes.flat:
            for p in ax.patches:
                height = p.get_height()
                ax.text(
                    p.get_x() + p.get_width() / 2,
                    height,
                    f"{height:.1f}",
                    ha="center",
                    va="bottom",
                    fontsize=12
                )

        plt.title(title)
        plt.tight_layout()
        plt.savefig(filepath)
        plt.close()








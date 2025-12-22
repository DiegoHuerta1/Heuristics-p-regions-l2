import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os



class ResultPlotter:
    """ 
    Class for plotting results of heuristic experiments (batch runs).
    """

    def __init__(self, heuristics: list[str], df: pd.DataFrame, plot_folder: str, name: str):
        """  
        Args:
            df (pd.DataFrame): DataFrame with results of heuristics.
                Format used for BatchRunner results (internally, with relative gap).
                Columns: ID, N, K, ...
                ID = Name__K__Repetition
            heuristics (list[str]): List of heuristic names.
            plot_folder (str): Folder to save plots.
            name (str): Name prefix for the plots.
        """
        self.name: str = name
        self.df: pd.DataFrame = df
        self.df = self.df.copy()
        self.df["K"] = pd.Categorical(self.df["K"].to_numpy())
        self.heuristics: list[str] = heuristics
        self.plot_folder: str = plot_folder

        # Particular folders
        self.f_folder: str = plot_folder + "f/"
        self.time_folder: str = plot_folder + "time/"
        self.rel_gap_folder: str = plot_folder + "relative_gap/"
        os.makedirs(self.f_folder, exist_ok=True)
        os.makedirs(self.time_folder, exist_ok=True)
        os.makedirs(self.rel_gap_folder, exist_ok=True)

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
    
    # Main visualization function -----------------------------

    def make_all_plots(self):

        # Boxplots of f 
        self.make_boxplot(self.df_melted_f, "Objective", self.f_folder)
        # Boxplots of time
        self.make_boxplot(self.df_melted_time, "Time", self.time_folder)
        # Relative gap
        self.make_barplot_with_text(self.df_melted_rel_gap, "Relative Gap", self.rel_gap_folder)


    # Individual plot functions ---------------------------------

    def make_boxplot(self, df_plot: pd.DataFrame, variable: str, folder: str):
        """ 
        Make a boxplot for the given data.
        """
        plt.figure(figsize=(12, 6))
        sns.boxplot(data = df_plot, x="Heuristic", y= variable, hue = "K")
        plt.title(f"{variable} distribution - {self.name}")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(folder + f"{self.name}__{variable}.png")
        plt.close()

    def make_barplot_with_text(self, df_plot: pd.DataFrame, variable: str, folder: str):
        """ 
        Make a barplot with text annotations for the given data.
        The bars are grouped by K.
        """
        df_plot = df_plot.copy()
        df_plot["K"] = df_plot["K"].astype(str)

        g = sns.catplot(df_plot, x="Heuristic", y= variable, kind="bar", hue = "K", 
                        height=6, aspect=2, errorbar=None)
        
        # Move legend outside (right)
        g.legend.set_bbox_to_anchor((1.05, 0.5)) # type: ignore
        g.legend.set_title("K") # type: ignore

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

        plt.title(f"{variable} distribution - {self.name}")
        plt.savefig(folder + f"{self.name}__{variable}.png", bbox_inches='tight')
        plt.close()








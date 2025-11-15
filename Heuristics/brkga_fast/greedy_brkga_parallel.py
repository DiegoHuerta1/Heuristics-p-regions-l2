import igraph
import numpy as np

from ..utils import generate_dissimilarity_matrix

import time
import pandas as pd
import matplotlib.pyplot as plt
from multiprocessing import Pool, shared_memory


# Importa las funciones del worker desde el archivo limpio
from .brkga_worker import _init_parallel_worker, _run_parallel_worker
# Importa también tu lógica de fitness/decode
from .brkga_worker import decode


class Greedy_BRKGA_parallel:

    def __init__(self, adj_graph_or_dict: igraph.Graph | dict,
                 num_regions: int,
                 dissimilarity_matrix: np.ndarray | None = None, **kwargs):
        """
        Args:
            num_regions (int): Number of regions to create
            dissimilarity_matrix (np.ndarray | None, optional): Matrix with euclidean squared distances.

        kwargs include:
            population_size (int)
            elite_fraction (float)
            mutant_fraction (float)
            crossover_rate (float)
            max_generations (int)
            tolerance_generations (int)
            max_time (int)
            seed (int | None)
        """

        self.N: int  # number of nodes
        self.n: int  # chromosome length
        self.K: int = num_regions
        self.adjacency: dict[int, list[int]]
        self.dissimilarity_matrix: np.ndarray
        # control the adjacency
        if isinstance(adj_graph_or_dict, igraph.Graph):
            self.N = adj_graph_or_dict.vcount()                 
            self.adjacency = {v: adj_graph_or_dict.neighbors(v) for v in range(self.N)}
            self.dissimilarity_matrix = generate_dissimilarity_matrix(adj_graph_or_dict)
        else:
            self.N = len(adj_graph_or_dict)                
            self.adjacency = adj_graph_or_dict
            assert dissimilarity_matrix is not None, "If adjacency dict is provided, dissimilarity matrix must be provided too."
            self.dissimilarity_matrix = dissimilarity_matrix
        # set other attributes
        self.num_pairs = self.N*(self.N - 1)//2   
        self.n = self.num_pairs + self.N     

        # ------------
        # GET PARAMETERS FROM KWARGS
        population_size = kwargs.get("population_size", 200)
        elite_fraction = kwargs.get("elite_fraction", 0.2)
        mutant_fraction = kwargs.get("mutant_fraction", 0.2)
        crossover_rate = kwargs.get("crossover_rate", 0.7)
        max_generations = kwargs.get("max_generations", 200)
        tolerance_generations = kwargs.get("tolerance_generations", 100)
        max_time = kwargs.get("max_time", 3600)
        seed = kwargs.get("seed", None)
        verbose = kwargs.get("verbose", False)
        self.number_of_processors = kwargs.get("number_of_processors", 5)

        # SET BRKGA PARAMETERS       
        if isinstance(population_size, float):
            self.p = int(population_size * self.n)
        else:
            self.p = population_size
        assert 0 < elite_fraction < 0.5, "Elite fraction must be in (0, 0.5)"
        self.p_e = int(self.p * elite_fraction)
        assert 0 < mutant_fraction < 1, "Mutant fraction must be in (0, 1)"
        assert elite_fraction + mutant_fraction < 1, "Elite and mutan fractions must add up to less than 1"
        self.p_m = int(self.p * mutant_fraction) 
        self.offspring_size = self.p - self.p_e - self.p_m
        assert 0.5 < crossover_rate < 1, "Crossover rate must be in (0.5, 1)"
        self.ro_e = crossover_rate
        self.max_generations = max_generations
        self.tolerance_generations = tolerance_generations
        self.max_time = max_time
        self.seed = seed
        self.verbose = verbose
        self.evolution_stats = {}

    # ------------------------------------------
    # UTILITY METHODS FOR THE BRKGA

    def generate_chromosome_array(self, number_of_chromosomes: int)-> np.ndarray:
        return np.random.rand(number_of_chromosomes, self.n)

    def parametrized_uniform_crossover(self, elite_parent: np.ndarray,
                                     non_elite_parent: np.ndarray) -> np.ndarray:
        random_variables = np.random.rand(self.n)
        return np.where(random_variables <= self.ro_e, elite_parent, non_elite_parent)

    def generate_offspring(self, population) -> np.ndarray:
        offspring = []
        for _ in range(self.offspring_size):
            # select two parents: one elite and one non-elite
            elite_parent = population[np.random.randint(0, self.p_e)]
            non_elite_parent = population[np.random.randint(self.p_e, self.p)]
            # create the child
            child = self.parametrized_uniform_crossover(elite_parent, non_elite_parent)
            offspring.append(child)
        return np.array(offspring)
    
    def sort_population(self, population: np.ndarray,
                        fitness_values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        sorted_indices = np.argsort(fitness_values)
        population = population[sorted_indices]
        fitness_values = fitness_values[sorted_indices]
        return population, fitness_values

    # ------------------------
    # Utils for plots 

    def print_generation_info(self, fitness_values: np.ndarray, idx: int):
        """
        Print information about the current generation
        """
        if self.verbose:
            _min = fitness_values.min()
            _mean = fitness_values.mean()
            _std = fitness_values.std()
            print(f"Generation {idx}: Best fitness = {_min:.6f}. Mean fitness = {_mean:.6f}. Std = {_std:.6f}")

    def compute_statistics(self, fitness_values: np.ndarray) -> dict:
        """
        Compute statistics from the fitness values of the population
        """
        return {
            "mean": fitness_values.mean(),
            "std": fitness_values.std(),
            "min": fitness_values.min(),
            "q10": np.quantile(fitness_values, 0.10),
            "q25": np.quantile(fitness_values, 0.25),
            "median": np.quantile(fitness_values, 0.50),
            "q75": np.quantile(fitness_values, 0.75),
            "q90": np.quantile(fitness_values, 0.90),
            "elite_cutoff": np.quantile(fitness_values, self.p_e/self.p) # elite quantile
        }
    
    def print_statistics(self):
        """
        Print the statistics of the evolution
        """
        print(f"Best fitness: {self.evolution_stats['best_fitness']:4f}")
        print(f"Execution time: {self.evolution_stats['time']:4f} seconds")
        print(f"Last generation: {self.evolution_stats['population_stats'].index.max()}")
        diffs = self.evolution_stats['population_stats']['min'].round(4).diff() < 0
        print(f"Best solution found on iteration: {diffs[diffs].index.max() if diffs.any() else 0}")

    def plot_evolution(self, image_path: str | None = None):
        """
        Plot the evolution of the population statistics
        Saves the plot if image_path is provided,
        otherwise shows it on screen.
        """

        df = self.evolution_stats["population_stats"]
        if df.empty:
            print("No statistics to plot.")
            return

        fig, ax = plt.subplots(figsize=(10, 4))
        # Interquartile range (25th to 75th percentile)
        ax.fill_between(df.index, df['q25'], df['q75'], color='blue', alpha=0.3, label='25–75% quantile')
        # Mean
        ax.plot(df.index, df['mean'], color='black', linestyle='--', label='Mean')
        # Median 
        ax.plot(df.index, df['median'], color='blue', label='Median')
        # Elite quantile
        ax.plot(df.index, df['elite_cutoff'], color='red', linestyle='--', label=f'Elite Cutoff ({100 * self.p_e/self.p:.0f}% quantile)')
        # Min
        ax.plot(df.index, df['min'], label=f"Minimum ({df['min'].iloc[-1]:.2f})", color='red')

        ax.set_title('Population Statistics')
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Fitness')
        ax.legend(
            loc='upper left',
            bbox_to_anchor=(1.02, 1),
            borderaxespad=0,
            frameon=False
        )
        plt.grid(True)
        if image_path is not None:
            plt.savefig(image_path, bbox_inches='tight')
        else:
            plt.show()
        plt.close()

    # ------------------------
    # Evolution

    def run(self):
        """
        Main method to evolve a population of chromosomes

        Saves a dictionary of results in self.evolution_stats
        This is a dictionary with the results:
            - best_chromosome: Best chromosome found
            - best_solution: Best solution found (decoded chromosome)
            - best_fitness: Fitness of the best solution
            - population_stats: Population statistics over generations
            - time: Execution time
        """
        population_statistics = []
        
        # set random seed and start time
        if self.seed is not None:
            np.random.seed(self.seed)
        start_time = time.time()

        # --- 1. PREPARACIÓN DE MEMORIA COMPARTIDA (SHM) ---
        
        # 1a. Datos Estáticos (Matriz Grande)
        static_data = self.dissimilarity_matrix
        static_shm = shared_memory.SharedMemory(create=True, size=static_data.nbytes)
        static_view_main = np.ndarray(static_data.shape, dtype=static_data.dtype, buffer=static_shm.buf)
        static_view_main[:] = static_data[:]
        
        # 1b. Datos Dinámicos (Población)
        pop_shape = (self.p, self.n)
        pop_dtype = np.float64
        pop_nbytes = int(np.prod(pop_shape) * np.dtype(pop_dtype).itemsize)
        pop_shm = shared_memory.SharedMemory(create=True, size=pop_nbytes)
        pop_view_main = np.ndarray(pop_shape, dtype=pop_dtype, buffer=pop_shm.buf)

        # 1c. Argumentos para el inicializador del worker
        # (Todos son objetos pequeños y seguros)
        init_args = (
            self.N,
            self.num_pairs,
            self.K,
            self.adjacency,
            pop_shm.name,
            pop_shape,
            pop_dtype,
            static_shm.name,
            static_data.shape,
            static_data.dtype
        )

        # --- 2. INICIO DEL POOL PERSISTENTE ---
        try:
            # Asegúrate de usar 'spawn' en tu main.py (if __name__ == "__main__":)
            # mp.set_start_method("spawn")
            with Pool(
                processes=self.number_of_processors,
                initializer=_init_parallel_worker,
                initargs=init_args
            ) as pool:
                
                # --- Generación 0 ---
                population = self.generate_chromosome_array(self.p)
                pop_view_main[:] = population[:] # Escribir en SHM
                # ¡La llamada al map solo envía índices int!
                fitness_values = np.array(pool.map(_run_parallel_worker, range(self.p)))

                # Sort population and save statistics
                population, fitness_values = self.sort_population(population, fitness_values)
                population_statistics.append(self.compute_statistics(fitness_values))
                self.print_generation_info(fitness_values, 0)

                # Control the generation loop 
                best_fitness = fitness_values.min()
                generations_without_improvement = 0

                # --- Bucle Principal (Generaciones 1+) ---
                # Main loop (generations 1 - max_generations)
                for idx in range(1, self.max_generations + 1):

                    # Create offspring and mutants
                    offspring = self.generate_offspring(population)
                    mutants = self.generate_chromosome_array(self.p_m)

                    # Escribir en SHM (¡rápido!)
                    pop_view_main[self.p_e:self.p_e+self.offspring_size] = offspring
                    pop_view_main[self.p_e+self.offspring_size:] = mutants
                    pop_view_main[:self.p_e] = population[:self.p_e] # Élites

                    # Evaluar solo los nuevos (solo se envían ints)
                    range_to_eval = range(self.p_e, self.p)
                    new_fitness = np.array(pool.map(_run_parallel_worker, range_to_eval))

                    # Actualizar (local)
                    fitness_values[self.p_e:] = new_fitness
                    population = pop_view_main.copy() # Copia de SHM a local

                    # Sort population and save statistics
                    population, fitness_values = self.sort_population(population, fitness_values)
                    population_statistics.append(self.compute_statistics(fitness_values))
                    self.print_generation_info(fitness_values, idx)

                    # Evaluate the tolerance condition 
                    current_best_fitness = np.min(fitness_values)
                    if current_best_fitness + 1e-4 < best_fitness: # improvement!
                        generations_without_improvement = 0
                        best_fitness = current_best_fitness
                    else:                                          # no improvement :(
                        generations_without_improvement += 1
                    if generations_without_improvement >= self.tolerance_generations:
                        break

                    # Evaluate the time condition
                    elapsed_time = time.time() - start_time
                    if elapsed_time >= self.max_time:
                        break
        
                # Get best solution
                best_idx = np.argmin(fitness_values)
                best_fitness = fitness_values[best_idx]
                best_chromosome = population[best_idx]
                best_solution = decode(best_chromosome,
                                    self.dissimilarity_matrix,
                                        self.N,
                                        self.num_pairs,
                                        self.K,
                                        self.adjacency)
                # Store evolution statistics
                self.evolution_stats = {
                    "best_chromosome": best_chromosome,
                    "best_solution": best_solution,
                    "best_fitness": float(best_fitness),
                    "population_stats": pd.DataFrame(population_statistics),
                    "time": time.time() - start_time
                }

        finally:
            # --- 3. LIMPIEZA DE MEMORIA COMPARTIDA ---
            # ¡Muy importante!
            pop_shm.close()
            pop_shm.unlink()
            static_shm.close()
            static_shm.unlink()


















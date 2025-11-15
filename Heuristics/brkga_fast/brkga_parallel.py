
import igraph
import numpy as np
from ..utils import generate_dissimilarity_matrix, l2_objective_function_diss_matrix
from functools import partial

from .parallel_procesor import ParallelMatrixProcessor

import time
import pandas as pd
import matplotlib.pyplot as plt


# ----------------------------------------------------------------------------------------------
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
        verbose = kwargs.get("verbose", True)

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
            elite_parent = population[np.random.randint(0, self.p_e)]
            non_elite_parent = population[np.random.randint(self.p_e, self.p)]
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

        try:
            # Initialize population (generation 0)
            population = self.generate_chromosome_array(self.p)
            # Function to evaluate in each chromosome
            F_func = partial(chromosome_fitness,
                            N = self.N, num_pairs = self.num_pairs,
                            K= self.K, adjacency= self.adjacency)
            # Número de cromosomas que vamos a pasar al procesador
            n_parallel = self.p - self.p_e
            # Divide la población en dos grupos
            parallel_population = population[:n_parallel]
            sequential_population = population[n_parallel:]
            # Procesar en paralelo
            processor = ParallelMatrixProcessor(parallel_population, self.dissimilarity_matrix,
                                                func= F_func, n_workers=5, chunk_size=10)
            fitness_parallel = processor.ejecutar()
            # Procesar secuencialmente
            fitness_sequential = np.array([F_func(ch, self.dissimilarity_matrix) for ch in sequential_population])
            # Concatenar resultados
            fitness_values = np.concatenate((fitness_parallel, fitness_sequential))

            # Sort population and save statistics
            population, fitness_values = self.sort_population(population, fitness_values)
            population_statistics.append(self.compute_statistics(fitness_values))
            self.print_generation_info(fitness_values, 0)

            # Control the generation loop 
            best_fitness = fitness_values.min()
            generations_without_improvement = 0

            # Main loop (generations 1 - max_generations)
            for idx in range(1, self.max_generations + 1):

                # Create offspring and mutants
                offspring = self.generate_offspring(population)
                mutants = self.generate_chromosome_array(self.p_m)
                new_individuals = np.vstack((offspring, mutants))
                # Compute fitness of new individuals
                processor.reemplazar_A(new_individuals)
                new_fitness = processor.ejecutar()

                # Update population
                population = np.vstack((population[:self.p_e], new_individuals))
                fitness_values = np.concatenate((fitness_values[:self.p_e], new_fitness))

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
        finally:
            processor.cleanup()
        # Store evolution statistics
        self.evolution_stats = {
            "best_chromosome": best_chromosome,
            "best_solution": best_solution,
            "best_fitness": float(best_fitness),
            "population_stats": pd.DataFrame(population_statistics),
            "time": time.time() - start_time
        }



# --------------------------------------------------------
# COMPUTE FITNESS 

def chromosome_fitness(chromosome: np.ndarray, 
                dissimilarity_matrix: np.ndarray,
                N: int, num_pairs: int,
                K: int,
                adjacency: dict[int, list[int]]) -> float:
    solution = decode(chromosome, dissimilarity_matrix, N, num_pairs, K, adjacency)
    return l2_objective_function_diss_matrix(solution, dissimilarity_matrix)

# --------------------------------------------------------
# DECODE

def decode(chromosome: np.ndarray, 
           dissimilarity_matrix: np.ndarray,
           N: int, num_pairs: int,
           K: int,
           adjacency: dict[int, list[int]]) -> dict[int, list[int]]:
    """
    Decode a chromosome into a solution.
    
    Greedy Decoder
    """

    # Obtain the dissimilarit matrix induced by the chromosome
    matrix_d = vec_to_sym(chromosome[:num_pairs], N) * dissimilarity_matrix
    # Get K seed nodes from the second part (lowest values)
    seed_nodes = np.argsort(chromosome[num_pairs:])[:K]
    
    # Keep track of assigned nodes
    assigned_nodes: set[int] = set(seed_nodes)

    # Start Partition with seeds 
    P: dict[int, list[int]] = {(idx+1): [int(seed)] for idx, seed in enumerate(seed_nodes)}
    # Start R_k with zeros
    R_k: dict[int, float] = {k: 0.0 for k in P.keys()}

    # Get feasible elements, and their evaluation under the greedy function
    feasible_elements_g: dict[tuple[int, int], float] = get_feasible_elements_greedy(P, 
                                                                                     adjacency,
                                                                                     assigned_nodes,
                                                                                     matrix_d,
                                                                                     R_k)

    # Create solution while there are feasible elements
    while feasible_elements_g:

        # Get the element with lowest evaluation
        v_star, k_star = min(feasible_elements_g, key=lambda e: feasible_elements_g[e])

        # Compute the future value of R_k_star after making the assignement
        future_R_k_star = compute_future_R_k(v_star, k_star, matrix_d, P, R_k)

        # Remove elements that assign v_star to other regions
        feasible_elements_g = {e: val for e, val in feasible_elements_g.items() if e[0] != v_star}
        # Update greedy evaluations of elements that assign to k_star
        for (v, k) in feasible_elements_g.keys():
            if k == k_star:
                feasible_elements_g[(v, k)] = update_greedy_eval(v, k, feasible_elements_g[(v, k)],
                                                                        v_star, future_R_k_star,
                                                                        matrix_d, P, R_k)

        # Update the partition and R_k
        P[k_star].append(v_star)
        R_k[k_star] = future_R_k_star
        assigned_nodes.add(v_star)

        # Get new feasible elements and their evaluations
        new_feasible_elements_g = get_new_feasible_elements_greedy(v_star, k_star,
                                                                    feasible_elements_g.keys(),
                                                                    matrix_d,
                                                                    adjacency, assigned_nodes,
                                                                    P, R_k)
        feasible_elements_g.update(new_feasible_elements_g)

    return P

# --------------------------------------------------------
# PLAIN FUNCTIONS
    
def vec_to_sym(vector: np.ndarray, N : int) -> np.ndarray:
    matrix = np.zeros((N, N))
    upper_indices = np.triu_indices(N, k=1)
    matrix[upper_indices] = vector
    matrix += matrix.T
    return matrix

def get_feasible_elements_greedy(P: dict[int, list[int]],
                                adjacency: dict[int, list[int]],
                                assigned_nodes: set[int],  
                                matrix_d: np.ndarray,
                                R_k: dict[int, float]
                                ) -> dict[tuple[int, int], float]:
    # Compute all feasible elements 
    # {(v, k) | (∄h ∈ [K] : v ∈ Ph) ∧ (∃u ∈ N(v) : u ∈ Pk)}
    feasible_elements: list[tuple] = []
    # iterate on asigned nodes
    for k, P_k in P.items():
        for u in P_k:
            # iterate on unnasigned neighbors
            for v in adjacency[u]:
                if v not in assigned_nodes:
                    # save the element
                    feasible_elements.append((v, k))

    # Evaluate all feasible elements under the greedy function
    feasible_elements_g: dict[tuple[int, int], float] = {}
    for (v, k) in feasible_elements:
        feasible_elements_g[(v, k)] = evaluate_greedy_element(v, k, matrix_d, P, R_k)
    return feasible_elements_g

def evaluate_greedy_element(v: int, k: int, matrix_d: np.ndarray,
                            P: dict[int, list[int]], R_k: dict[int, float]) -> float:
    sum_dissimilarities = sum(matrix_d[v, i] for i in P[k])
    evaluation = 1/(len(P[k]) + 1) * (sum_dissimilarities - R_k[k])
    return evaluation

def update_greedy_eval(v: int, k: int, old_eval: float, 
                        v_star: int, new_R_k: float,
                        matrix_d: np.ndarray,
                        P: dict[int, list[int]], R_k: dict[int, float]) -> float:
    n_k = len(P[k])
    new_eval = 1/(n_k + 2) * ((n_k + 1)*old_eval + R_k[k] - new_R_k + matrix_d[v, v_star])
    return new_eval

def compute_future_R_k(v: int, k: int, matrix_d: np.ndarray,
                       P: dict[int, list[int]], R_k: dict[int, float]) -> float:
    n_k = len(P[k])
    return 1/(n_k + 1) * (n_k * R_k[k] + sum(matrix_d[v, i] for i in P[k]))

def get_new_feasible_elements_greedy(v_star: int, k_star: int,
                                     current_feasible, matrix_d: np.ndarray,
                                     adjacency: dict[int, list[int]], assigned_nodes: set[int],
                                     P: dict[int, list[int]], R_k: dict[int, float]) -> dict[tuple[int, int], float]:
    # Compute feasible elements 
    # {(v, k∗) : (v ∈ N (v∗)) ∧ (∄h ∈ [K] : v ∈ Ph) ∧ ((v, k∗) /∈ F)}
    new_feasible_elements: list[tuple] = []
    for v in adjacency[v_star]:
        if v not in assigned_nodes and (v, k_star) not in current_feasible:
            new_feasible_elements.append((v, k_star))
    # Evaluate all new feasible elements under the greedy function
    new_feasible_elements_g: dict[tuple[int, int], float] = {}
    for (v, k) in new_feasible_elements:
        new_feasible_elements_g[(v, k)] = evaluate_greedy_element(v, k, matrix_d, P, R_k)
    return new_feasible_elements_g





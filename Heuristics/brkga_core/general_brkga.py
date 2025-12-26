import igraph
import numpy as np
import time
import pandas as pd
import matplotlib.pyplot as plt
from multiprocessing import Pool
from typing import Callable, Iterable

from .parallel_processor import ParallelMatrixProcessor
from .utils import EvolutionStats, Fit_Seq, Fit_Par, Decoder


class BRKGAPRegions():
    """
    Base class for Biased Random-Key Genetic Algorithm (BRKGA) for the P-regions problem.

    Arguments:    
        name - Name for this particular BRKGA
        chromosome_length - Length of each chromosome array
        init_worker_func - Function for parallel workers
        init_args - args for the init_worker_func function
        fitness_seq - Function to compute fitness sequentially
        fitness_parallel - Function to compute the fitness on each worker 
        decoder_func - Function to decode a chromosome into a partition
        chromosome_generator - Function for custom chromosomes on generation 0
        dissimilarity_matrix - Dissimilarity matrix for the specific instance

    Extra arguments:
        population_size - Absolute size (int) or relative to chromosome_length (float). Default 200
        elite_fraction - Fraction of elite individuals in the population. Default 0.2
        mutant_fraction - Fraction of mutants in the population. Default 0.2
        crossover_rate - Crossover rate for offspring generation. Default 0.7
        max_generations - Maximum number of generations to evolve. Default 200
        tolerance_generations - Number of generations without improvement to stop. Default 100
        max_time - Maximum time (in seconds) to run the evolution. Default 3600
        parallel - Run in parallel (True) or sequentially (False). Default True
        num_workers - Number of parallel workers (if parallel is True). Default 4
        seed - Random seed for reproducibility. Default None
        verbose - Verbosity level (0 = silent, 1 = minimal, 2 = detailed). Default 0


    """

    def __init__(self, name: str, chromosome_length: int,
                 init_worker_func: Callable, init_args: Iterable,
                 fitness_seq: Fit_Seq, 
                 fitness_parallel: Fit_Par,
                 decoder_func: Decoder,
                 chromosome_generator: Callable[[int], np.ndarray],
                 parallel: bool,
                 dissimilarity_matrix: np.ndarray, **kwargs):
        
        # Parameters for this specific BRKGA implementation
        self.name: str = name
        self.init_worker_func: Callable = init_worker_func
        self.init_args: Iterable = init_args
        self.fitness_seq: Fit_Seq = fitness_seq
        self.fitness_parallel: Fit_Par = fitness_parallel
        self.decoder_func: Decoder = decoder_func  
        self.chromosome_generator: Callable[[int], np.ndarray] = chromosome_generator  

        # Population parameters
        self.n: int = chromosome_length
        self.p: int
        population_size = kwargs.get("population_size", 200)
        if isinstance(population_size, float):
            self.p = int(population_size * self.n)
        else:
            self.p = population_size
        elite_fraction = kwargs.get("elite_fraction", 0.2)
        assert 0 < elite_fraction < 0.5, "Elite fraction must be in (0, 0.5)"
        self.p_e: int = int(self.p * elite_fraction)
        mutant_fraction = kwargs.get("mutant_fraction", 0.2)
        assert 0 < mutant_fraction < 1, "Mutant fraction must be in (0, 1)"
        assert elite_fraction + mutant_fraction < 1, "Elite and mutan fractions must add up to less than 1"
        self.p_m: int = int(self.p * mutant_fraction)  
        self.offspring_size: int = self.p - self.p_e - self.p_m

        # Select one run method
        if parallel:
            self.run = self.run_parallel
        else:
            self.run = self.run_sequential

        # Aditional parameters
        self.dissimilarity_matrix = dissimilarity_matrix
        crossover_rate = kwargs.get("crossover_rate", 0.7)
        assert 0.5 < crossover_rate < 1, "Crossover rate must be in (0.5, 1)"
        self.ro_e: float = crossover_rate
        self.max_generations = kwargs.get("max_generations", 200)
        self.tolerance_generations = kwargs.get("tolerance_generations", 100)
        self.max_time =  kwargs.get("max_time", 3600)
        self.seed = kwargs.get("seed", None)
        self.num_workers =  kwargs.get("num_workers", 4) 
        self.verbose = kwargs.get("verbose", 0)
        self.evolution_stats: EvolutionStats

        # Describe BRKGA
        self.print_general_info(f"\n{self.name} BRKGA for P-Regions Problem", level = 1)
        self.print_general_info(f"Chromosome of length: {self.n}", level = 1)
        self.print_general_info(f"Population of size: {self.p}", level = 1)
        self.print_general_info(f"\tElite: {self.p_e}", level = 2)
        self.print_general_info(f"\tMutants: {self.p_m}", level = 2)
        self.print_general_info(f"\tOffspring: {self.offspring_size}", level = 2)

    # ------------------------------------------
    # Utility methods for brkga dynamics

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
        fitness_values = fitness_values.round(decimals=8)
        sorted_indices = np.argsort(fitness_values)
        population = population[sorted_indices]
        fitness_values = fitness_values[sorted_indices]
        return population, fitness_values

    # ------------------------
    # Utils for plots 

    def print_general_info(self, message: str, level: int):
        if self.verbose >= level:
            print(message, flush=True)

    def print_generation_info(self, fitness_values: np.ndarray, idx: int):
        """
        Print information about the current generation
        """
        _min = fitness_values.min()
        _mean = fitness_values.mean()
        _median = np.median(fitness_values)
        self.print_general_info(f"\tGeneration {idx}: Best fitness = {_min:.6f}. Mean = {_mean:.6f}. Median = {_median:.6f}", level = 2)

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

    # ------------------------
    # Evolution

    def run_parallel(self):
        """
        Main method to evolve a population of chromosomes in parallel
        """
        population_statistics = []
        
        # set random seed and start time
        if self.seed is not None:
            np.random.seed(self.seed)
            self.print_general_info(f"Random seed set to {self.seed}", level = 2)
        start_time = time.time()

        # Initialize pool of parallel workers
        self.print_general_info(f"{self.name} BRKGA Evolution ({self.num_workers}-parallel)", level = 1)
        n_parallel = self.p - self.p_e
        chunk_size =  int(np.ceil(n_parallel / self.num_workers))
        with Pool(processes = self.num_workers,
                  initializer = self.init_worker_func, 
                  initargs= self.init_args) as pool: 
            processor = None 
            self.print_general_info("Pool created", level = 2)
            self.print_general_info(f"Evolution with {self.num_workers} processors. Chunks of size {chunk_size}", level = 2)

            try:
                
                # Initialize population (generation 0)

                # Part 1 (custom chromosome, evaluated secuentially)
                size_pop1 = self.p_e
                pop1 = self.chromosome_generator(size_pop1)
                fitnes_pop1 = np.array([self.fitness_seq(c, self.dissimilarity_matrix) for c in pop1])
                # Part 2 (normal chromosome, evaluated in parallel)
                size_pop2 = n_parallel
                pop2 = self.generate_chromosome_array(size_pop2)
                processor = ParallelMatrixProcessor(pop2 ,
                                                    self.dissimilarity_matrix,
                                                    func= self.fitness_parallel, 
                                                    pool= pool,
                                                    chunk_size=chunk_size)
                fitnes_pop2 = processor.execute()
                # Merge
                population = np.vstack((pop1, pop2))
                fitness_values = np.concatenate((fitnes_pop1, fitnes_pop2))

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
                    processor.replace_A(new_individuals)
                    new_fitness = processor.execute()

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

                self.print_general_info("Evolution finished", level = 1) 
                # Get best solution
                best_idx = np.argmin(fitness_values)
                best_fitness = fitness_values[best_idx]
                best_chromosome = population[best_idx]
                best_solution = self.decoder_func(best_chromosome, self.dissimilarity_matrix)
                # Store evolution statistics
                self.evolution_stats = {
                    "best_chromosome": best_chromosome,
                    "best_solution": best_solution,
                    "best_fitness": float(best_fitness),
                    "population_stats": pd.DataFrame(population_statistics),
                    "time": time.time() - start_time
                }

            # Clean shared memory
            finally:
                if processor:
                        processor.cleanup()


    def run_sequential(self):
        """
        Main method to evolve a population of chromosomes in parallel
        """
        population_statistics = []
        
        # set random seed and start time
        if self.seed is not None:
            np.random.seed(self.seed)
        start_time = time.time()
                
        # Initialize population (generation 0)
        self.print_general_info(f"{self.name} BRKGA Evolution (sequential)", level = 1)

        # Part 1 (custom chromosome)
        size_pop1 = self.p_e
        pop1 = self.chromosome_generator(size_pop1)
        fitnes_pop1 = np.array([self.fitness_seq(c, self.dissimilarity_matrix) for c in pop1])
        # Part 2 (normal chromosome)
        size_pop2 = self.p - self.p_e
        pop2 = self.generate_chromosome_array(size_pop2)
        fitnes_pop2 = np.array([self.fitness_seq(c, self.dissimilarity_matrix) for c in pop2])
        # Merge
        population = np.vstack((pop1, pop2))
        fitness_values = np.concatenate((fitnes_pop1, fitnes_pop2))

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
            new_fitness = np.array([self.fitness_seq(c, self.dissimilarity_matrix) for c in new_individuals])

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
    
        self.print_general_info("Evolution finished", level = 1)
        # Get best solution
        best_idx = np.argmin(fitness_values)
        best_fitness = fitness_values[best_idx]
        best_chromosome = population[best_idx]
        best_solution = self.decoder_func(best_chromosome, self.dissimilarity_matrix)
        # Store evolution statistics
        self.evolution_stats = {
            "best_chromosome": best_chromosome,
            "best_solution": best_solution,
            "best_fitness": float(best_fitness),
            "population_stats": pd.DataFrame(population_statistics),
            "time": time.time() - start_time
        }


    # --------------------------------------------------------------------
    # Post run methods

    def print_statistics(self):
        """
        Print the statistics of the evolution
        """
        print(f"{self.name} BRKGA results:")
        print(f"\tBest fitness: {self.evolution_stats['best_fitness']:4f}")
        print(f"\tExecution time: {self.evolution_stats['time']:4f} seconds")
        print(f"\tLast generation: {self.evolution_stats['population_stats'].index.max()}")
        diffs = self.evolution_stats['population_stats']['min'].round(4).diff() < 0
        print(f"\tBest solution found on iteration: {diffs[diffs].index.max() if diffs.any() else 0}")

    def plot_evolution(self, image_path: str | None = None):
        """
        Plot the evolution of the population statistics
        Saves the plot if image_path is provided,
        otherwise shows it on screen.
        """

        df = self.evolution_stats["population_stats"]
        if df.empty:
            self.print_general_info("No statistics to plot.", level = 1)
            return

        _, ax = plt.subplots(figsize=(10, 4))
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




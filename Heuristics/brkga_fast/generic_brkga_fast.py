from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
from typing import Any
from multiprocessing import Pool

# Set up for parallel evaluation
NUMBER_OF_PROCESSORS = 4
_shared_obj = None
def _init_worker(obj): 
    global _shared_obj
    _shared_obj = obj
def _worker(chromosome):  
    return _shared_obj.chromosome_fitness(chromosome)


class GenericBRKGA_fast(ABC):
    """
    Base class for Biased Random-Key Genetic Algorithm (BRKGA).
    Subclasses must implement the decode() method.
    """

    def __init__(self, chromosome_length: int,
                 population_size: int | float = 100,
                 elite_fraction: float = 0.2, mutant_fraction: float = 0.2,
                 crossover_rate: float = 0.7, max_generations: int = 100,
                 tolerance_generations: int = 100, max_time: int = 3600,
                 seed: int | None = None, verbose: bool = False): 
        """
        Class constructor

        Args:
            chromosome_length (int): Number of genes in each chromosome.
            population_size (int | float): Number of chromosomes in population. Defaults to 100
                If it is a float, the population size will be int(population_size * chromosome_length)
            elite_fraction (float): Fraction of the population that is elite. Defaults to 0.2.
            mutant_fraction (float): Fraction of the population that is mutant. Defaults to 0.7.
            crossover_rate (float): Biased crossover rate for the elite parent. Defaults to 0.7.
            max_generations (int): Maximum number of generations. Defaults to 100.
            tolerance_generations (int): Stop after some generations without improvement in best fitness. Defaults to 100.
            max_time (int): Stop after the defined number of seconds. Defaults to 3600.
            seed (int | None): Control randomness. Defaults to None.
            verbose (bool): Whether to print progress messages. Defaults to False.
        """

        # set general attributes
        self.n = chromosome_length
        if isinstance(population_size, float):
            self.p = int(population_size * self.n)
        else:
            self.p = population_size
        assert 0 < elite_fraction < 0.5, "Elite fraction must be in (0, 0.5)"
        self.p_e = int(population_size * elite_fraction) 
        assert 0 < mutant_fraction < 1, "Mutant fraction must be in (0, 1)"
        assert elite_fraction + mutant_fraction < 1, "Elite and mutan fractions must add up to less than 1"
        self.p_m = int(population_size * mutant_fraction)  
        self.offspring_size = self.p - self.p_e - self.p_m
        assert 0.5 < crossover_rate < 1, "Crossover rate must be in (0.5, 1)"
        self.ro_e = crossover_rate

        # set algorithm parameters
        self.max_generations = max_generations
        self.tolerance_generations = tolerance_generations
        self.max_time = max_time
        self.seed = seed
        self.verbose = verbose
        # update after evolution
        self.evolution_stats = {}

    # ------------------------
    # Abstract methods

    @abstractmethod
    def decode(self, chromosome) -> Any:
        """Decode a chromosome into a solution."""
        pass

    @abstractmethod
    def chromosome_fitness(self, chromosome) -> Any:
        """Evaluate the fitness of a chromosome"""
        pass

    # ------------------------
    # Utils for evolution 

    def generate_chromosome_array(self, number_of_chromosomes: int)-> np.ndarray:
        """
        Generates an array of chromosomes (matrix)
        Each chromosome is a row of the matrix.
        A chromosome is a vector in [0, 1]^n

        Args:
            number_of_chromosomes (int): number of chromosomes to generate

        Returns:
            np.ndarray: A matrix of shape (number_of_chromosomes, self.n)
        """
        return np.random.rand(number_of_chromosomes, self.n)
    

    def parametrized_uniform_crossover(self, elite_parent: np.ndarray,
                                     non_elite_parent: np.ndarray) -> np.ndarray:
        """
        Parametrized uniform crossover, biased on the elite parent

        Args:
            elite_parent (np.ndarray): Elite parent chromosome
            non_elite_parent (np.ndarray): Non-elite parent chromosome

        Returns:
            np.ndarray: Offspring chromosome
        """
        random_variables = np.random.rand(self.n)
        return np.where(random_variables <= self.ro_e, elite_parent, non_elite_parent)

    def generate_offspring(self, population) -> np.ndarray:
        """
        Generate offspring from the current population

        Returns:
            np.ndarray: Matrix (num_offspring, self.n)
        """
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
        """
        Sort population and fitness values according to fitness values (ascending)
        """
        sorted_indices = np.argsort(fitness_values)
        population = population[sorted_indices]
        fitness_values = fitness_values[sorted_indices]
        return population, fitness_values

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

        # Start pool
        with Pool(
            processes = NUMBER_OF_PROCESSORS,
            initializer = _init_worker,
            initargs=(self,)  
        ) as pool:

            # Initialize population (generation 0)
            population = self.generate_chromosome_array(self.p)
            fitness_values_list = pool.map(_worker, population)
            fitness_values = np.array(fitness_values_list)

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
                new_fitness_list = pool.map(_worker, new_individuals)
                new_fitness = np.array(new_fitness_list)

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
            best_solution = self.decode(best_chromosome)

            # Store evolution statistics
            self.evolution_stats = {
                "best_chromosome": best_chromosome,
                "best_solution": best_solution,
                "best_fitness": float(best_fitness),
                "population_stats": pd.DataFrame(population_statistics),
                "time": time.time() - start_time
            }

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



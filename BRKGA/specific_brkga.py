import igraph
import numpy as np
import itertools

from .generic_brkga import GenericBRKGA
from .utils import generate_dissimilarity_matrix, l2_objective_function_diss_matrix, compute_P_names


# ----------------------------------------------------------------------------------------------
class MST_BRKGA(GenericBRKGA):
    """
    Minimum Spanning Tree Decoder for BRKGA
    """

    def __init__(self, graph: igraph.Graph, num_regions: int,
                 dissimilarity_matrix: np.ndarray | None = None, **kwargs):
        """
        Args:
            graph (igraph.Graph): Graph instance to perform regionalization
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

        # set general attributes
        self.G = graph.copy()
        self.M = graph.ecount() # number of edges
        self.d = 2*self.M       # chromosome length
        self.K = num_regions

        # dissimilarity matrix
        if dissimilarity_matrix is None:
            self.dissimilarity_matrix = generate_dissimilarity_matrix(graph)
        else:
            self.dissimilarity_matrix = dissimilarity_matrix

        # define dissimilarity weights on the graph edges
        edges = graph.get_edgelist()
        diss_weights = [self.dissimilarity_matrix[i, j] for i, j in edges]
        self.diss_weights = np.array(diss_weights)

        # call parent constructor
        super().__init__(chromosome_length = self.d, **kwargs)


    def decode(self, chromosome) -> dict[int, list[int]]:
        """
        Decode a chromosome into a solution.
        
        MST Decoder
        """

        # Get two weights from the chromosome 
        w_minus = chromosome[:self.M] * self.diss_weights
        w_plus = chromosome[self.M:]

        # Get edges from a minimum spanning tree using w_minus
        mst_edges_id = self.G.spanning_tree(weights = w_minus, return_tree = False)

        # Drop the first K-1 edges (considering the weights in w_plus)
        mst_edges_id.sort(key = lambda e: w_plus[e], reverse = True)
        final_edges_id = mst_edges_id[(self.K - 1):]

        # Remove all edges from the graph, excpet the final edges
        edges_2_remove = set(range(self.M)) - set(final_edges_id)
        G_copy = self.G.copy()
        G_copy.delete_edges(edges_2_remove)

        # Make the partition using the connected components
        components = G_copy.connected_components()
        P = {idx+1: nodes for idx, nodes in enumerate(components)}
        return P
    

    def chromosome_fitness(self, chromosome) -> float:
        """
        Evaluate the fitness of a chromosome
        
        Decodes the chromosome into a solution and evaluates under the objective function
        """
        solution = self.decode(chromosome)
        return l2_objective_function_diss_matrix(solution, self.dissimilarity_matrix)


# ----------------------------------------------------------------------------------------------
class ST_BRKGA(GenericBRKGA):
    """
    Shortest Path Decoder for BRKGA
    """

    def __init__(self, graph: igraph.Graph, num_regions: int,
                 dissimilarity_matrix: np.ndarray | None = None, **kwargs):
        """
        Args:
            graph (igraph.Graph): Graph instance to perform regionalization
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

        # set general attributes
        self.G = graph.copy()
        self.N = graph.vcount()   # number of nodes
        self.M = graph.ecount()   # number of edges
        self.d = self.M + self.N  # chromosome length
        self.K = num_regions

        # dissimilarity matrix
        if dissimilarity_matrix is None:
            self.dissimilarity_matrix = generate_dissimilarity_matrix(graph)
        else:
            self.dissimilarity_matrix = dissimilarity_matrix

        # define dissimilarity weights on the graph edges
        edges = graph.get_edgelist()
        diss_weights = [self.dissimilarity_matrix[i, j] for i, j in edges]
        self.diss_weights = np.array(diss_weights)

        # call parent constructor
        super().__init__(chromosome_length = self.d, **kwargs)


    def decode(self, chromosome) -> dict[int, list[int]]:
        """
        Decode a chromosome into a solution.
        
        ST Decoder
        """

        # Get edge weigths from the first part
        edge_weigths = chromosome[:self.M] * self.diss_weights
        # Get K seed nodes from the second part (lowest values)
        seed_nodes = np.argsort(chromosome[self.M:])[:self.K]

        # run dijkstra from each seed node
        dist_from_seeds = self.G.distances(source = seed_nodes,
                                           weights = edge_weigths,
                                           algorithm="dijkstra")
        
        # Assign each node to the index of the closest seed 
        P = {k: [] for k in range(1, self.K + 1)}
        for node in range(self.N):
            # dict of distances from seeds to the node
            distances = {(idx+1): dist_from_seeds[idx][node]
                         for idx in range(self.K)}
            # select the closest 
            k_star = min(distances, key = lambda k: (distances[k], k))
            P[k_star].append(node)

        return P
    

    def chromosome_fitness(self, chromosome) -> float:
        """
        Evaluate the fitness of a chromosome
        
        Decodes the chromosome into a solution and evaluates under the objective function
        """
        solution = self.decode(chromosome)
        return l2_objective_function_diss_matrix(solution, self.dissimilarity_matrix)
    

# ----------------------------------------------------------------------------------------------
class Greedy_BRKGA(GenericBRKGA):
    """
    Greedy Decoder for BRKGA
    """

    def __init__(self, graph: igraph.Graph, num_regions: int,
                 dissimilarity_matrix: np.ndarray | None = None, **kwargs):
        """
        Args:
            graph (igraph.Graph): Graph instance to perform regionalization
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

        # set general attributes
        self.G = graph.copy()
        self.N = graph.vcount()                   # number of nodes
        self.num_pairs = self.N*(self.N - 1)//2   # number of pairs
        self.d = self.num_pairs + self.N          # chromosome total length
        self.K = num_regions

        # dissimilarity matrix
        if dissimilarity_matrix is None:
            self.dissimilarity_matrix = generate_dissimilarity_matrix(graph)
        else:
            self.dissimilarity_matrix = dissimilarity_matrix

        # call parent constructor
        super().__init__(chromosome_length = self.d, **kwargs)


    def vec_to_sym(self, vector: np.ndarray) -> np.ndarray:
        """ 
        Transforms a vector of length self.num_pairs into a symmetric matrix self.N x self.N
        """
        matrix = np.zeros((self.N, self.N))
        upper_indices = np.triu_indices(self.N, k=1)
        matrix[upper_indices] = vector
        matrix += matrix.T
        return matrix
    

    def get_feasible_elements_greedy(self) -> dict[tuple[int, int], float]:
        """
        Get feasible elements and their evaluation under the greedy function.

        A feasible element is a pair (v, k) where v is a node and k is a region.
        """

        # Compute all feasible elements 
        # {(v, k) | (∄h ∈ [K] : v ∈ Ph) ∧ (∃u ∈ N(v) : u ∈ Pk)}
        feasible_elements: list[tuple] = []
        # iterate on asigned nodes
        for k, P_k in self.P.items():
            for u in P_k:
                # iterate on unnasigned neighbors
                for v in self.G.neighbors(u):
                    if v not in self.assigned_nodes:
                        # save the element
                        feasible_elements.append((v, k))

        # Evaluate all feasible elements under the greedy function
        feasible_elements_g: dict[tuple[int, int], float] = {}
        for (v, k) in feasible_elements:
            feasible_elements_g[(v, k)] = self.evaluate_greedy_element(v, k)
        return feasible_elements_g
    
    
    def evaluate_greedy_element(self, v: int, k: int) -> float:
        """ 
        Evaluate greedy function for a given element (v, k).

        Proposition 2
        """
        sum_dissimilarities = sum(self.matrix_d[v, i] for i in self.P[k])
        evaluation = 1/(len(self.P[k]) + 1) * (sum_dissimilarities - self.R_k[k])
        return evaluation
    

    def update_greedy_eval(self, v: int, k: int, old_eval: float, 
                            v_star: int, new_R_k: float) -> float:
        """ 
        Takes an assigned element (v, k) with current value under the greedy function old eval
        Returns the new evaluation that will result after selecting the element (v_star, k)
        new_R_k is the value that R_k will have after selecting the element (v_star, k)

        Proposition 3
        """
        n_k = len(self.P[k])
        new_eval = 1/(n_k + 2) * ((n_k + 1)*old_eval + self.R_k[k] - new_R_k + self.matrix_d[v, v_star])
        return new_eval
    
    
    def compute_future_R_k(self, v: int, k: int) -> float:
        """ 
        Compute the value R_k(P')
        where P' is the result of applying (v, k) to P

        Equation 13
        """
        n_k = len(self.P[k])
        return 1/(n_k + 1) * (n_k * self.R_k[k] + sum(self.matrix_d[v, i] for i in self.P[k]))
    

    def get_new_feasible_elements_greedy(self, v_star: int, k_star: int,
                                         current_feasible) -> dict[tuple[int, int], float]:
        """ 
        Get new feasible elements after assigning v_star to k_star.
        current_feasible is the list of currently feasible elements (v, k)
        """

        # Compute feasible elements 
        # {(v, k∗) : (v ∈ N (v∗)) ∧ (∄h ∈ [K] : v ∈ Ph) ∧ ((v, k∗) /∈ F)}
        new_feasible_elements: list[tuple] = []
        for v in self.G.neighbors(v_star):
            if v not in self.assigned_nodes and (v, k_star) not in current_feasible:
                new_feasible_elements.append((v, k_star))

        # Evaluate all new feasible elements under the greedy function
        new_feasible_elements_g: dict[tuple[int, int], float] = {}
        for (v, k) in new_feasible_elements:
            new_feasible_elements_g[(v, k)] = self.evaluate_greedy_element(v, k)
        return new_feasible_elements_g


    def decode(self, chromosome) -> dict[int, list[int]]:
        """
        Decode a chromosome into a solution.
        
        Greedy Decoder
        """

        # Obtain the dissimilarit matrix induced by the chromosome
        self.matrix_d = self.vec_to_sym(chromosome[:self.num_pairs]) * self.dissimilarity_matrix
        # Get K seed nodes from the second part (lowest values)
        seed_nodes = np.argsort(chromosome[self.num_pairs:])[:self.K]
        
        # Keep track of assigned nodes
        self.assigned_nodes: set[int] = set(seed_nodes)

        # Start Partition with seeds 
        self.P: dict[int, list[int]] = {(idx+1): [int(seed)] for idx, seed in enumerate(seed_nodes)}
        # Start R_k with zeros
        self.R_k: dict[int, float] = {k: 0.0 for k in self.P.keys()}

        # Get feasible elements, and their evaluation under the greedy function
        feasible_elements_g: dict[tuple[int, int], float] = self.get_feasible_elements_greedy()

        # Create solution while there are feasible elements
        while feasible_elements_g:
            # self.check(feasible_elements_g)

            # Get the element with lowest evaluation
            v_star, k_star = min(feasible_elements_g, key=lambda e: feasible_elements_g[e])

            # Compute the future value of R_k_star after making the assignement
            future_R_k_star = self.compute_future_R_k(v_star, k_star)

            # Remove elements that assign v_star to other regions
            feasible_elements_g = {e: val for e, val in feasible_elements_g.items() if e[0] != v_star}
            # Update greedy evaluations of elements that assign to k_star
            for (v, k) in feasible_elements_g.keys():
                if k == k_star:
                    feasible_elements_g[(v, k)] = self.update_greedy_eval(v, k, feasible_elements_g[(v, k)],
                                                                          v_star, future_R_k_star)

            # Update the partition and R_k
            self.P[k_star].append(v_star)
            self.R_k[k_star] = future_R_k_star
            self.assigned_nodes.add(v_star)

            # Get new feasible elements and their evaluations
            new_feasible_elements_g = self.get_new_feasible_elements_greedy(v_star, k_star,
                                                                            feasible_elements_g.keys())
            feasible_elements_g.update(new_feasible_elements_g)

        return self.P
    

    def check(self, feasible_elements_g):
        """ 
        Sanity check (testing only)
        """
        f_P = l2_objective_function_diss_matrix(self.P, self.matrix_d)
        assert np.isclose(sum(self.R_k.values()), f_P) # R_k working!
        for (v, k), val in feasible_elements_g.items():
            P_prima = {key:value.copy() for key,value in self.P.items()}
            P_prima[k].append(v)
            f_prima = l2_objective_function_diss_matrix(P_prima, self.matrix_d)
            assert np.isclose(f_prima - f_P, val)  # greedy function working! 
    

    def chromosome_fitness(self, chromosome) -> float:
        """
        Evaluate the fitness of a chromosome
        
        Decodes the chromosome into a solution and evaluates under the objective function
        """
        solution = self.decode(chromosome)
        return l2_objective_function_diss_matrix(solution, self.dissimilarity_matrix)
    









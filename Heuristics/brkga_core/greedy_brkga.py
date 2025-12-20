import igraph
import numpy as np
from functools import partial

from .general_brkga import BRKGAPRegions
from .Decoders.greedy import greedy_decoder, greedy_fitness
from .utils import build_adjacency_arrays, P_from_array_to_dict, P_Dict
from ..utils import generate_dissimilarity_matrix



_WORKER_ADJ_OFFSETS = None
_WORKER_ADJ_NEIG = None


def _init_worker(offsets: np.ndarray, neighbors: np.ndarray):
    global _WORKER_ADJ_OFFSETS
    global _WORKER_ADJ_NEIG
    _WORKER_ADJ_OFFSETS = offsets
    _WORKER_ADJ_NEIG = neighbors


def chromosome_fitness_wrapper(chromosome: np.ndarray, 
                               dissimilarity_matrix: np.ndarray,
                               N: int, K: int, rank: int, break_point: int) -> float:
    return greedy_fitness(chromosome, dissimilarity_matrix,
                          N, K, rank, break_point,
                          _WORKER_ADJ_OFFSETS, _WORKER_ADJ_NEIG) # type: ignore



class Greedy_BRKGA(BRKGAPRegions):

    def __init__(self, graph: igraph.Graph, num_regions: int, 
                 dissimilarity_matrix: np.ndarray | None = None, **kwargs):
        
        # Dissimilarity matrix
        if dissimilarity_matrix is None:
            dissimilarity_matrix = generate_dissimilarity_matrix(graph)

        # Greedy breakpoint
        rank = kwargs.get("rank", 1)
        num_nodes = graph.vcount()
        break_point = rank * num_nodes

        # Transform adjacency to arrays                             
        adjacency = {v: graph.neighbors(v) for v in range(num_nodes)}
        self.adj_offsets, self.adj_neighbors = build_adjacency_arrays(adjacency, num_nodes)

        # Pool arguments
        init_worker_func = _init_worker
        init_args = (self.adj_offsets, self.adj_neighbors)

        # Sequential fitness
        fitness_seq = partial(greedy_fitness, 
                              N = num_nodes, K = num_regions,
                              rank = rank, break_point = break_point,
                              adj_offsets = self.adj_offsets,
                              adj_neighbors = self.adj_neighbors)

        # Parallel fitness
        fitness_parallel = partial(chromosome_fitness_wrapper,
                                   N = num_nodes, K = num_regions,
                                   rank = rank, break_point = break_point)

        # Decoder
        def decoder_func(chromosome: np.ndarray, diss_matrix: np.ndarray) -> P_Dict:
            p, _ = greedy_decoder(chromosome, diss_matrix,
                                    N = num_nodes, K = num_regions,
                                    rank = rank, break_point = break_point,
                                    adj_offsets = self.adj_offsets,
                                    adj_neighbors = self.adj_neighbors)
            return P_from_array_to_dict(p, num_regions)
        
        # Create custom chromosomes
        def chromosome_generator(size_pop: int) -> np.ndarray:
            pop_first_half = np.ones((size_pop, break_point))
            pop_second_half = np.random.rand(size_pop, num_nodes)
            pop = np.hstack((pop_first_half, pop_second_half))
            return pop


        # Parent constructor
        super().__init__(chromosome_length = break_point + num_nodes,
                         init_worker_func = init_worker_func,
                         init_args = init_args,
                         fitness_seq = fitness_seq,
                         fitness_parallel = fitness_parallel,
                         decoder_func = decoder_func,
                         chromosome_generator = chromosome_generator,
                         dissimilarity_matrix = dissimilarity_matrix, **kwargs)
        












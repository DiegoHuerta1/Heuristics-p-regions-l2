import igraph
import numpy as np
from functools import partial
import igraph

from .general_brkga import BRKGAPRegions
from .Decoders.st import st_decoder, st_fitness
from .utils import P_Dict
from ..utils import generate_dissimilarity_matrix


_WORKER_GRAPH = None
_WORKER_DISS_WEIGHTS = None


def _init_worker(num_nodes: int, edges: np.ndarray, diss_weights: np.ndarray):
    global _WORKER_GRAPH
    global _WORKER_DISS_WEIGHTS
    _WORKER_GRAPH  = igraph.Graph(n = num_nodes, edges = edges.tolist())
    _WORKER_DISS_WEIGHTS = diss_weights



def chromosome_fitness_wrapper(chromosome: np.ndarray, 
                               dissimilarity_matrix: np.ndarray,
                               num_edges: int, K: int) -> float:
    return st_fitness(chromosome, dissimilarity_matrix, num_edges, K, 
                       _WORKER_GRAPH, _WORKER_DISS_WEIGHTS) # type: ignore



class ST_BRKGA(BRKGAPRegions):

    def __init__(self, graph: igraph.Graph, num_regions: int, 
                 dissimilarity_matrix: np.ndarray | None = None, **kwargs):
        
        # Graph basics
        num_nodes = graph.vcount()
        num_edges = graph.ecount()
        edges = graph.get_edgelist()

        # Dissimilarity matrix
        if dissimilarity_matrix is None:
            dissimilarity_matrix = generate_dissimilarity_matrix(graph)

        # Dissimilarity weights
        diss_weights = [dissimilarity_matrix[i, j] for i, j in edges]
        diss_weights = np.array(diss_weights)

        # Pool arguments
        init_worker_func = _init_worker
        init_args = (num_nodes, np.array(edges), diss_weights)

        # Sequential fitness
        fitness_seq = partial(st_fitness, 
                              num_edges = num_edges, K = num_regions,
                              G = graph, diss_weights = diss_weights)

        # Parallel fitness
        fitness_parallel = partial(chromosome_fitness_wrapper,
                                   num_edges = num_edges, K = num_regions)

        # Decoder
        def decoder_func(chromosome: np.ndarray, diss_matrix: np.ndarray) -> P_Dict:
            P = st_decoder(chromosome, diss_matrix,
                           num_edges = num_edges, K = num_regions,
                           G = graph, diss_weights = diss_weights)
            return P
        
        # Create custom chromosomes (nothing spetial)
        def chromosome_generator(size_pop: int) -> np.ndarray:
            pop_first_half = np.ones((size_pop, num_edges))
            pop_second_half = np.random.rand(size_pop, num_nodes)
            pop = np.hstack((pop_first_half, pop_second_half))
            return pop


        # Parent constructor
        super().__init__("Shortest-Path", chromosome_length = num_edges + num_nodes,
                         init_worker_func = init_worker_func,
                         init_args = init_args,
                         fitness_seq = fitness_seq,
                         fitness_parallel = fitness_parallel,
                         decoder_func = decoder_func,
                         chromosome_generator = chromosome_generator,
                         dissimilarity_matrix = dissimilarity_matrix, **kwargs)
        












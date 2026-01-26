import igraph
import numpy as np
from functools import partial
import igraph

from ..general_brkga import BRKGAPRegions
from ..Decoders.st import st_fitness, st_decoder
from ..utils import build_adjacency_arrays_with_edges, P_from_array_to_dict, P_Dict
from ...utils import generate_dissimilarity_matrix


_WORKER_DISS_WEIGHTS = None
_WORKER_ADJ_OFFSETS = None
_WORKER_ADJ_NEIG = None
_WORKER_ADJ_EDGES = None


def _init_worker(diss_weights: np.ndarray, adj_offsets: np.ndarray,
                adj_neighbors: np.ndarray, adj_edges: np.ndarray):
    global _WORKER_DISS_WEIGHTS
    global _WORKER_ADJ_OFFSETS
    global _WORKER_ADJ_NEIG
    global _WORKER_ADJ_EDGES
    _WORKER_DISS_WEIGHTS = diss_weights
    _WORKER_ADJ_OFFSETS = adj_offsets
    _WORKER_ADJ_NEIG = adj_neighbors
    _WORKER_ADJ_EDGES = adj_edges



def chromosome_fitness_wrapper(chromosome: np.ndarray, diss_matrix: np.ndarray,
                               num_nodes: int, num_edges: int, K: int) -> float:
    return st_fitness(chromosome, diss_matrix, num_nodes, num_edges, K,
                      _WORKER_DISS_WEIGHTS, _WORKER_ADJ_OFFSETS, # type: ignore
                      _WORKER_ADJ_NEIG, _WORKER_ADJ_EDGES) # type: ignore



class ST_BRKGA(BRKGAPRegions):

    def __init__(self, graph: igraph.Graph, num_regions: int, 
                 dissimilarity_matrix: np.ndarray | None = None, **kwargs):
        
        # Graph basics
        num_nodes = graph.vcount()
        num_edges = graph.ecount()
        edges: np.ndarray = np.array(graph.get_edgelist())

        # Dissimilarity matrix
        if dissimilarity_matrix is None:
            dissimilarity_matrix = generate_dissimilarity_matrix(graph)

        # Dissimilarity weights
        diss_weights = [dissimilarity_matrix[i, j] for i, j in edges]
        diss_weights = np.array(diss_weights)

        # Get adjacency arrays                             
        adj_offsets, adj_neighbors, adj_edges = build_adjacency_arrays_with_edges(graph)

        # Pool arguments
        init_worker_func = _init_worker
        init_args = (diss_weights, adj_offsets, adj_neighbors, adj_edges)

        # Sequential fitness
        fitness_seq = partial(st_fitness, 
                              num_nodes = num_nodes, num_edges = num_edges, K = num_regions,
                              diss_weights = diss_weights, adj_offsets = adj_offsets,
                              adj_neighbors = adj_neighbors, adj_edges = adj_edges)

        # Parallel fitness
        fitness_parallel = partial(chromosome_fitness_wrapper,
                                   num_nodes = num_nodes, num_edges = num_edges, K = num_regions)

        # Decoder
        def decoder_func(chromosome: np.ndarray, diss_matrix: np.ndarray) -> P_Dict:
            p = st_decoder(chromosome, diss_matrix,
                           num_nodes = num_nodes, num_edges = num_edges, K = num_regions,
                           diss_weights = diss_weights, adj_offsets = adj_offsets,
                           adj_neighbors = adj_neighbors, adj_edges = adj_edges)
            return P_from_array_to_dict(p, num_regions)
        
        # Create custom chromosomes (nothing spetial)
        def chromosome_generator(size_pop: int) -> np.ndarray:
            pop_first_half = np.ones((size_pop, num_edges))
            pop_second_half = np.random.rand(size_pop, num_nodes)
            pop = np.hstack((pop_first_half, pop_second_half))
            return pop

        # Select parallel vs sequential
        parallel_arg = kwargs.get("parallel_brkga", True)
        parallel: bool
        if isinstance(parallel_arg, str) and parallel_arg.lower() == "auto":
            parallel = True if num_nodes >= 100 else False
        elif isinstance(parallel_arg, bool):
            parallel = parallel_arg
        else:
            parallel = False


        # Parent constructor
        super().__init__("Shortest-Path", chromosome_length = num_edges + num_nodes,
                         init_worker_func = init_worker_func,
                         init_args = init_args,
                         fitness_seq = fitness_seq,
                         fitness_parallel = fitness_parallel,
                         decoder_func = decoder_func,
                         chromosome_generator = chromosome_generator,
                         parallel = parallel,
                         dissimilarity_matrix = dissimilarity_matrix, **kwargs)
        












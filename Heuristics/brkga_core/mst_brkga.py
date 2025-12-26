import igraph
import numpy as np
from functools import partial

from .general_brkga import BRKGAPRegions
from .Decoders.mst import mst_fitness, mst_decoder
from .utils import P_Dict, P_from_array_to_dict
from ..utils import generate_dissimilarity_matrix


_WORKER_NUM_NODES = None
_WORKER_EDGES = None
_WORKER_DISS_WEIGHTS = None


def _init_worker(num_nodes: int, edges: np.ndarray, diss_weights: np.ndarray):
    global _WORKER_NUM_NODES
    global _WORKER_EDGES
    global _WORKER_DISS_WEIGHTS
    _WORKER_NUM_NODES = num_nodes
    _WORKER_EDGES = edges
    _WORKER_DISS_WEIGHTS = diss_weights



def chromosome_fitness_wrapper(chromosome: np.ndarray, 
                               dissimilarity_matrix: np.ndarray,
                               num_edges: int, K: int) -> float:
    return mst_fitness(chromosome, dissimilarity_matrix, num_edges, K, 
                       _WORKER_NUM_NODES, _WORKER_EDGES, _WORKER_DISS_WEIGHTS) # type: ignore



class MST_BRKGA(BRKGAPRegions):

    def __init__(self, graph: igraph.Graph, num_regions: int, 
                 dissimilarity_matrix: np.ndarray | None = None, **kwargs):
        
        # Graph basics
        num_nodes: int = graph.vcount()
        num_edges: int = graph.ecount()
        edges: np.ndarray = np.array(graph.get_edgelist())

        # Dissimilarity matrix
        if dissimilarity_matrix is None:
            dissimilarity_matrix = generate_dissimilarity_matrix(graph)

        # Dissimilarity weights
        diss_weights: np.ndarray = np.array([dissimilarity_matrix[i, j] for i, j in edges])

        # Pool arguments
        init_worker_func = _init_worker
        init_args = (num_nodes, edges, diss_weights)

        # Sequential fitness
        fitness_seq = partial(mst_fitness, 
                              num_edges = num_edges, K = num_regions,
                              num_nodes = num_nodes, edges = edges,
                              diss_weights = diss_weights)

        # Parallel fitness
        fitness_parallel = partial(chromosome_fitness_wrapper,
                                   num_edges = num_edges, K = num_regions)

        # Decoder
        def decoder_func(chromosome: np.ndarray, diss_matrix: np.ndarray) -> P_Dict:
            p = mst_decoder(chromosome,
                            num_edges = num_edges, K = num_regions,
                            num_nodes = num_nodes, edges = edges,
                            diss_weights = diss_weights)
            return P_from_array_to_dict(p, num_regions)
        
        # Create custom chromosomes (nothing spetial)
        def chromosome_generator(size_pop: int) -> np.ndarray:
            return np.random.rand(size_pop, num_edges * 2)
        
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
        super().__init__("Minimum-Spanning-Tree", chromosome_length = num_edges * 2,
                         init_worker_func = init_worker_func,
                         init_args = init_args,
                         fitness_seq = fitness_seq,
                         fitness_parallel = fitness_parallel,
                         decoder_func = decoder_func,
                         chromosome_generator = chromosome_generator,
                         parallel = parallel,
                         dissimilarity_matrix = dissimilarity_matrix, **kwargs)
        












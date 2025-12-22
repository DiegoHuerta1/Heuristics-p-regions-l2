import igraph
import numpy as np
from hypothesis import strategies as st
from hypothesis import given, settings
from hypothesis.extra.numpy import arrays
from Heuristics.utils import generate_dissimilarity_matrix

# pytest -v Heuristics/Test/test_decoders.py

# Utils -----------------------------------------------

def equal_partitions(P1: dict, P2: dict, K: int):
    """  
    Check if two partitions (dict representation, 1 idx) are equal
    """
    for k in range(1, K + 1):
        if set(P1[k]) != set(P2[k]):
            return False 
    return True


# Define some possible instances (Instances_Mexico)
IDS = [str(n).zfill(2) for n in range(1, 32)]


def get_instance_data(id: str) -> tuple[igraph.Graph, list[int], np.ndarray]:
    """  
    Get all information about an instance
    Using Instances_Mexico
    """
    # Read graph
    instance_path = f"./Instances_Mexico/{id}.pkl"
    with open(instance_path, "rb") as f:
        graph = igraph.Graph.Read_Pickle(f)
    # Options for K
    K_options = list(range(2, min(graph.vcount() - 1, 10) ))
    # Diss matrix
    diss_matrix = generate_dissimilarity_matrix(graph)
    return graph, K_options, diss_matrix


@st.composite
def draw_all_info(draw, length_fn, rank=0):
    """  
    Get instance information, number of regions and chromosome array.
    The length_fn specifies the length of the chromosome
    """
    # Draw instance
    id_ = draw(st.sampled_from(IDS))
    graph, K_opts, diss = get_instance_data(id_)

    # Draw K
    K = draw(st.sampled_from(K_opts))

    # Draw a population
    pop_size = 5
    chromosome_length = length_fn(graph, rank) 
    pop = draw(
        arrays(
            dtype = np.float64,
            shape=(pop_size, chromosome_length),
            elements = st.floats(0.01, 0.99),
        )
    )

    return id_, graph, diss, K, pop


# MST Test -----------------------------------------------

def mst_length(graph, rank):
    return 2 * graph.ecount()

@settings(deadline=5000)  
@given(data = draw_all_info(mst_length))
def test_mst(data):
    from ..brkga_core_deprecated.specific_brkga import MST_BRKGA as MST_BRKGA_old
    from ..brkga_core.mst_brkga import MST_BRKGA

    # Draw all elements
    _, graph, diss, num_regions, pop = data

    # Try both methods
    brkga_old = MST_BRKGA_old(graph, num_regions, diss)
    brkga = MST_BRKGA(graph, num_regions, diss)

    for c in pop:

        # Compare P
        P_old = brkga_old.decode(c)
        P_new = brkga.decoder_func(c, diss)
        assert equal_partitions(P_old, P_new, num_regions)
        # Compare f
        f_old = brkga_old.chromosome_fitness(c)
        f_new = brkga.fitness_seq(c, diss)
        assert np.isclose(f_old, f_new)


# ST Test -----------------------------------------------

def st_length(graph, rank):
    return graph.ecount() + graph.vcount()

@settings(deadline=5000)  
@given(data = draw_all_info(st_length))
def test_st(data):
    from ..brkga_core_deprecated.specific_brkga import ST_BRKGA as ST_BRKGA_old
    from ..brkga_core.st_brkga import ST_BRKGA

    # Draw all elements
    _, graph, diss, num_regions, pop = data

    # Try both methods
    brkga_old = ST_BRKGA_old(graph, num_regions, diss)
    brkga = ST_BRKGA(graph, num_regions, diss)

    for c in pop:

        # Compare P
        P_old = brkga_old.decode(c)
        P_new = brkga.decoder_func(c, diss)
        assert equal_partitions(P_old, P_new, num_regions)
        # Compare f
        f_old = brkga_old.chromosome_fitness(c)
        f_new = brkga.fitness_seq(c, diss)
        assert np.isclose(f_old, f_new)


# Greedy Test -----------------------------------------------

def greedy_length(graph, rank):
    return graph.vcount() * rank + graph.vcount()

@settings(deadline=10000)  
@given(rank=st.integers(min_value=1, max_value=5), data=st.data())
def test_greedy(rank, data):
    from ..brkga_core_deprecated.greedy_rank_decoder import chromosome_fitness, decode
    from ..brkga_core.greedy_brkga import Greedy_BRKGA

    # Draw all elements using the rank
    _, graph, diss, num_regions, pop = data.draw(
        draw_all_info(greedy_length, rank)
    )
    adj = {v: graph.neighbors(v) for v in range(graph.vcount())}

    # Just the new method is a BRKGA (ols is just functions)
    brkga = Greedy_BRKGA(graph, num_regions, diss, rank = rank)

    for c in pop:

        # Compare P
        P_old = decode(c, diss, N = graph.vcount(), rank = rank,
                        break_point = graph.vcount() * rank,
                        K = num_regions, adjacency= adj)
        P_new = brkga.decoder_func(c, diss)
        assert equal_partitions(P_old, P_new, num_regions)
        # Compare f
        f_old = chromosome_fitness(c, diss, N = graph.vcount(), rank = rank,
                                    break_point = graph.vcount() * rank,
                                    K = num_regions, adjacency= adj)
        f_new = brkga.fitness_seq(c, diss)
        assert np.isclose(f_old, f_new)



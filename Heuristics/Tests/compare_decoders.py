from Heuristics.utils import get_mexican_instance_data
import numpy as np
import time

# python -m Heuristics.Tests.compare_decoders


def main():
    instance = "07"
    num_regions = 10
    graph, _, diss_matrix = get_mexican_instance_data(instance)


    # MST BRKGA ------------------------------------------------
    from Heuristics.brkga_core_deprecated.specific_brkga import MST_BRKGA as MST_BRKGA_old
    from Heuristics.brkga_core.mst_brkga import MST_BRKGA

    # Draw a population
    size_pop = 1000
    n = 2 * graph.ecount()
    pop = np.random.rand(size_pop, n)

    # Old
    brkga_old = MST_BRKGA_old(graph, num_regions, diss_matrix)
    start_time_old  = time.time()
    fit_old = np.array([brkga_old.chromosome_fitness(c) for c in pop])
    time_old = time.time() - start_time_old

    # New
    brkga = MST_BRKGA(graph, num_regions, diss_matrix)
    start_time_new  = time.time()
    fit_new = np.array([brkga.fitness_seq(c, diss_matrix) for c in pop])
    time_new = time.time() - start_time_new

    # Compare
    print("MST BRKGA")
    print(f"Old time: {time_old:.4f} seconds")
    print(f"New time: {time_new:.4f} seconds")
    if np.allclose(fit_old, fit_new):
        print("Fitness values match!")


    # # ST BRKGA ------------------------------------------------
    # from Heuristics.brkga_core_deprecated.specific_brkga import ST_BRKGA as ST_BRKGA_old
    # from Heuristics.brkga_core.st_brkga import ST_BRKGA

    # # Draw a population
    # size_pop = 500
    # n = graph.ecount() + graph.vcount()
    # pop = np.random.rand(size_pop, n)

    # # Old
    # brkga_old = ST_BRKGA_old(graph, num_regions, diss_matrix)
    # start_time_old  = time.time()
    # fit_old = [brkga_old.chromosome_fitness(c) for c in pop]
    # time_old = time.time() - start_time_old

    # # New
    # brkga = ST_BRKGA(graph, num_regions, diss_matrix)
    # start_time_new  = time.time()
    # fit_new = [brkga.fitness_seq(c, diss_matrix) for c in pop]
    # time_new = time.time() - start_time_new

    # # Compare
    # print("ST BRKGA")
    # print(f"Old time: {time_old:.4f} seconds")
    # print(f"New time: {time_new:.4f} seconds")
    # if np.allclose(fit_old, fit_new):
    #   print("Fitness values match!")


    # # Greedy BRKGA ------------------------------------------------
    # from Heuristics.brkga_core_deprecated.greedy_rank_decoder import chromosome_fitness
    # from Heuristics.brkga_core.greedy_brkga import Greedy_BRKGA


    # # Draw a population
    # rank = 1
    # size_pop = 100
    # n = graph.vcount() * rank + graph.vcount()
    # pop = np.random.rand(size_pop, n)


    # # Old
    # adj = {v: graph.neighbors(v) for v in range(graph.vcount())}
    # start_time_old  = time.time()
    # fit_old = [chromosome_fitness(c, diss_matrix, N = graph.vcount(), rank = rank, 
    #                               break_point = graph.vcount() * rank, 
    #                               K = num_regions, adjacency= adj) for c in pop]
    # time_old = time.time() - start_time_old

    # # New
    # brkga = Greedy_BRKGA(graph, num_regions, diss_matrix, rank = rank)
    # start_time_new  = time.time()
    # fit_new = [brkga.fitness_seq(c, diss_matrix) for c in pop]
    # time_new = time.time() - start_time_new

    # # Compare
    # print("Greedy BRKGA")
    # print(f"Old time: {time_old:.4f} seconds")
    # print(f"New time: {time_new:.4f} seconds")
    # if np.allclose(fit_old, fit_new):
    #   print("Fitness values match!")



if __name__ == "__main__":
    main()








from Heuristics.utils import get_mexican_instance_data
import numpy as np
import time


instance = "20"
num_regions = 10
graph, _, diss_matrix = get_mexican_instance_data(instance)


# MST BRKGA ------------------------------------------------
from Heuristics.brkga_core_deprecated.specific_brkga import ST_BRKGA as ST_BRKGA_old
from Heuristics.brkga_core.st_brkga import ST_BRKGA


# Draw a population
np.random.seed(0)
size_pop = 500
n = graph.ecount() + graph.vcount()
pop = np.random.rand(size_pop, n)
c = pop[382]

# Old
brkga_old = ST_BRKGA_old(graph, num_regions, diss_matrix)
#P_old = brkga_old.decode(c)
fit_old = brkga_old.chromosome_fitness(c)
print(fit_old)

# New
brkga = ST_BRKGA(graph, num_regions, diss_matrix)
#P_new = brkga.decoder_func(c, diss_matrix)
fit_new = brkga.fitness_seq(c, diss_matrix) 
print(fit_new)












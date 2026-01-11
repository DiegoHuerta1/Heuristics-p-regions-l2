import igraph
import time
import numpy as np
from .ls_utils import PARTITION, LS_Stats
from .ls_utils import (obtener_funcion_H_plus_particion,
                       obtener_funcion_H_plus_particion_vecina,
                       obtener_fronteras_particion,
                       obtener_fronteras_particion_vecina,
                       get_posibles_movimientos,
                       funcion_objetivo_matriz_dist_complete,
                       funcion_objetivo_movimiento_matriz_dist,
                       hacer_movimiento)



def local_search_from_solution(grafo: igraph.Graph,
                               P0: PARTITION,
                               diss_matrix: np.ndarray) -> LS_Stats:
    
    '''
    Implementa busqueda local sobre una solucion existente.
    Utiliza la estrategia de la mejor mejora
    '''
    start_time = time.time()

    # inicializar P en la solucion inicial
    P = P0
    # obtenre esctructuras auxiliares
    conjuntos_cambio_P = obtener_funcion_H_plus_particion(grafo, P)
    fronteras_P = obtener_fronteras_particion(grafo, P, conjuntos_cambio_P)
    # ver su valor en la funcion objetivo
    f_P, f_P_regiones = funcion_objetivo_matriz_dist_complete(P, diss_matrix)
    # ir guardando un historial de las evaluaciones de la funcion
    historial_f = [f_P]

    # variables de control de la busqueda
    explorar = True

    while explorar:
        # el mejor movimiento es no hacer nada
        mejor_movimiento = None        
        f_best_N = f_P
        f_best_N_reg = f_P_regiones
        
        # tomar todos los movimientos posibles
        mov_posibles = get_posibles_movimientos(dict_fronteras_P= fronteras_P,
                                                dict_conjuntos_cambio_P= conjuntos_cambio_P)

        # explorar las posibilidades de movimientos
        while len(mov_posibles) > 0:
            mov_seleccionado = mov_posibles.pop()

            # ver la evaluacion de la funcion obejtivo en el vecino correspondiente
            f_P_prima_reg, f_P_prima = funcion_objetivo_movimiento_matriz_dist(P, f_P_regiones,
                                                                               mov_seleccionado,
                                                                               diss_matrix)

            # ver si es el mejor vecino hasta ahora
            if f_P_prima < f_best_N:
                f_best_N = f_P_prima
                f_best_N_reg = f_P_prima_reg
                mejor_movimiento = mov_seleccionado
                                
        # end while de explorar los movimientos
          
        # ver si ninugun movimiento fue mejor que quedarnos quietos
        if mejor_movimiento is None:
            explorar = False

        # si es que si hay mejor movimiento pasarse a el
        else:
            P = hacer_movimiento(P, mejor_movimiento)

            # actualizar estructuras auxiliares
            conjuntos_cambio_P = obtener_funcion_H_plus_particion_vecina(grafo, P,
                                                                         conjuntos_cambio_P,
                                                                         movimiento= mejor_movimiento)
            fronteras_P = obtener_fronteras_particion_vecina(grafo, P,
                                                             fronteras_P,
                                                             movimiento= mejor_movimiento,
                                                             conjuntos_cambio_vecino= conjuntos_cambio_P)


            # actualizar su evaluacion en la funcion
            f_P = f_best_N
            f_P_regiones = f_best_N_reg
                        
            # agregar la evaluacion a la funcion
            historial_f.append(f_P)

        # end else
        
    # end while principal
        
    # ya se tiene le optimo local
    # junto con su evaluacion en la funcion obejtivo
    # en P y f_P

    # ordenar lo que se quiere devovler
    resultados: LS_Stats = {
        "P": P,
        "f_P": f_P,
        "historial_f": historial_f,
        "time": time.time() - start_time
        }
    return resultados




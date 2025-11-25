import numpy as np
import time
import random
import itertools

# Busqueda local en una solucion inicial existente P0
def busqueda_local_desde_solucion_inicial(grafo, P0,
                                          matriz_distancias,
                                          tiempo_maximo= np.inf):
    '''
    Implementa busqueda local sobre una solucion existente

    Utiliza la estrategia de la mejor mejora

    '''
    # variable que indica si se llego a optimo local o no
    optimo_local = False

    # contar tiempo de ejecucion
    start_time = time.time()

    # ir guardando un historial de las evaluaciones de la funcion
    historial_f = []

    # inicializar P en la solucion inicial
    P = P0

    # obtener los conjuntos de cambio de esta solucion
    conjuntos_cambio_P = obtener_funcion_H_plus_particion(grafo, P)

    # obtener las fronteras de esta solucion
    fronteras_P = obtener_fronteras_particion(grafo, P, conjuntos_cambio_P)

    # ver su valor en la funcion objetivo, y contribucion por regiones
    f_P, f_P_regiones = funcion_objetivo_matriz_dist_complete(
        grafo, P, matriz_distancias)

    # agregar la evaluacion de la solucion inicial
    historial_f.append(f_P)

    # comenzar iteraciones de busqueda local
    explorar = True

    while explorar:

        # el mejor movimiento es no hacer nada
        mejor_movimiento = None
        # y pues lo mejor es quedarse
        f_best_N = f_P
        f_best_N_reg = f_P_regiones

        # tomar todos los movimientos posibles
        mov_posibles = get_posibles_movimientos(dict_fronteras_P=fronteras_P,
                                                dict_conjuntos_cambio_P=conjuntos_cambio_P)

        # explorar las posibilidades de movimientos
        while len(mov_posibles) > 0:

            # tomar un movimiento aleatorio y quitarlo de la lista
            random_index = random.randint(0, len(mov_posibles) - 1)
            mov_seleccionado = mov_posibles.pop(random_index)

            # ver la evaluacion de la funcion obejtivo
            # en el vecino correspondiente
            f_P_prima_reg, f_P_prima = funcion_objetivo_movimiento_matriz_dist(grafo, P, f_P_regiones,
                                                                               mov_seleccionado,
                                                                               matriz_distancias)

            # ver si es el mejor vecino hasta ahora
            if f_P_prima < f_best_N:

                # actualizar el mejor
                f_best_N = f_P_prima
                f_best_N_reg = f_P_prima_reg
                # guardar el movimiento, el mejor hasta ahora
                mejor_movimiento = mov_seleccionado

            # end if

        # en while de explorar los movimientos

        # ver si ninugun movimiento fue mejor que quedarnos quietos
        if mejor_movimiento is None:

            # se esta en optimo local
            optimo_local = True
            # ya se termino de explorar
            explorar = False

        # si es que si hay mejor movimiento
        # pasarse a el
        else:

            # actualizar los conjuntos de cambio del nuevo p
            conjuntos_cambio_P = obtener_funcion_H_plus_particion_vecina(grafo, P,
                                                                         conjuntos_cambio_P,
                                                                         movimiento=mejor_movimiento)
            # actualizar las fronteras del nuevo P
            fronteras_P = obtener_fronteras_particion_vecina(grafo, P,
                                                             fronteras_P,
                                                             movimiento=mejor_movimiento,
                                                             conjuntos_cambio_vecino=conjuntos_cambio_P)

            # aplicar el movimiento
            # para tomar el nuevo P
            P = hacer_movimiento(P, mejor_movimiento)
            # actualizar su evaluacion en la funcion
            f_P = f_best_N
            f_P_regiones = f_best_N_reg

            # agregar la evaluacion a la funcion
            historial_f.append(f_P)

        # end else

        # ver si ya se excedio el tiempo maximo
        if time.time() - start_time > tiempo_maximo:
            # no continuar con la busqueda
            explorar = False

    # end while principal

    # ya se tiene le optimo local
    # junto con su evaluacion en la funcion obejtivo
    # en P y f_P

    # ver cuanto tardo
    end_time = time.time()
    elapsed_time = end_time - start_time

    # ordenar lo que se quiere devovler
    resultados = {
        "P": P,
        "f_P": f_P,
        "tiempo": elapsed_time,
        "historial_f": historial_f,
        "optimo_local": optimo_local,
    }

    # devolver
    return resultados



# ---------------------------------------------------------------------------


# Funciones auxiliares del conjunto de cambio H^+


# devuelve el indice de la region del nodo v
# en la particion P
def funcion_h(v, P):

    # iterar en las regiones con su indice
    for idx_region, region in P.items():

        # si el nodo esta en esta region
        if v in region:

            # devolver el indice
            return idx_region

    # end for de las regiones
    # si el vertice no esta en ninguna region
    # indicar esto
    raise Exception(f"Nodo {v} no se encuentra en la particion")


# devuelve las etiquetas de las regiones de los vecinos de v
# en la particion P
def funcion_H(v, P, grafo):

    # tomar las etiquetas a devolver
    etiquetas_devolver = [funcion_h(u, P)
                          for u in grafo.neighbors(v)]

    # devolver como conjunto
    return set(etiquetas_devolver)


# devuelve las etiquetas de las regiones de los vecinos de v, sin la propia etiqueta
# en la particion P
# este es el conjunto de cambio de un vertice
def funcion_H_plus(v, P, grafo):

    # tomar H(v)
    H_v = funcion_H(v, P, grafo)

    # tomar h(v)
    h_v = funcion_h(v, P)

    # devolver la diferencia
    return H_v - {h_v}


# obtener el cojunto de cambio de todos los vertices en el grafo,
# usando la particion P
def obtener_funcion_H_plus_particion(grafo, P):
    '''
    Dada la particion P
    {i: nodos_en_i}

    Calcular el conjunto de cambio de todos los vertices
    {v: H+(v)}
    '''

    conjuntos_cambio = {node_v: funcion_H_plus(node_v, P, grafo)
                        for node_v in range(grafo.vcount())}

    return conjuntos_cambio


# ---------------------------------------------------------------------------


# Funciones auxiliares de frontera


# ver si un nodo se puede quitar de su region
# manteniendo factibilidad
def se_puede_quitar_nodo_region(grafo, P, i, v):
    '''
    grafo - grafo de igraph
    P - Particion P
    i - region de de P
    v - nodo v en P i


    Ve si el nodo v se puede quitar de su region (la region i)

    Devuelve:
        True si se puede quitar      (v not in C(G_i))
        False si no se puede quitar  (v in C(G_i))
    donde C(G_i) = {v en G tal que: G-v es disconexo o vacio}
    '''

    # tomar la subgrafica de solo nodos en P_i
    subgrafo_Gi = grafo.subgraph(P[i])

    # se quiere quitar v de este subgrafo
    # pero el indice cambia
    # se identifica a v por nombre

    # tomar el nombde de v
    nombre_v = grafo.vs[v]['name']

    # encotrar a v en la subgrafica
    # se hace buscandolo con su nombre
    v_subgrafo = subgrafo_Gi.vs.find(name=nombre_v).index

    # quitara v de la subgrafica
    subgrafo_Gi.delete_vertices(v_subgrafo)

    # ver si es que se rompio factibilidad
    if subgrafo_Gi.vcount() == 0 or not subgrafo_Gi.is_connected():

        # se vuelve disconexo o vacio, no se puede quitar
        return False

    # si no se devuelve False pues se devueelve True
    # v si se puede quitar de la subgrafica
    return True


# devuelve los nodos que estan en la frontera de la region i en la particion P
def funcion_frontera(grafo, P, i,
                     conjuntos_cambio_vertices):

    # primero filtrar los que tengan conjunto de cambio no vacio
    frontera_preliminar = [v for v in P[i]
                           if len(conjuntos_cambio_vertices[v]) > 0]

    # solo conservar los que se puedan quitar de esa ragion
    frontera = [v for v in frontera_preliminar
                if se_puede_quitar_nodo_region(grafo, P, i, v)]

    # devolver la frontera
    return frontera


# devuelde la frontera de todas las regiones en una particion
def obtener_fronteras_particion(grafo, P, conjuntos_cambio_vertices):
    '''
    Dada la particion P
    {i: nodos_en_i}

    Y los conjuntos de cambio de todos los vertices
    {v: H+(v)}

    Calcular frontera de cada region
    {i: nodos_en_frontera_de_i}
    '''

    fronteras_regiones = {indice_i: funcion_frontera(grafo, P, indice_i,
                                                     conjuntos_cambio_vertices)
                          for indice_i in P.keys()}

    return fronteras_regiones


# --------------------------------------------------------------------------------
# Funciones movimiento y vecinos
# Un movimiento es una tupla (i, v, j)
# signifca pasar el nodo v de la region i a la j


# devuelve una particion producto de
# aplicar el movimiento mov a la particion P
def hacer_movimiento(P, mov):

    # tomar los elementos del movimiento
    i = mov[0]
    v = mov[1]
    j = mov[2]

    # copiar la solucion P para hacer el vecino
    vecino_P = P.copy()

    # sacar el nodo v de la reigon P_i
    vecino_P[i] = [u for u in P[i] if u != v]

    # meter v a P_j
    vecino_P[j] = P[j] + [v]

    # devolver el vecino
    return vecino_P


# obtener posibles movimientos
# dadas las frontes y conjuntos de cambio de una particion
def get_posibles_movimientos(dict_fronteras_P,
                             dict_conjuntos_cambio_P):
    '''
    No se tiene una particion explicitamente,
    pero se tienen informacion sobre esta, se tiene

    1) Sus fronteras
    {i: nodos_en_frontera_de_i}

    2) sus conjuntos de cambio
    {v: H+(v)}


    Con esto se 
    obtiene una lista de tuplas
    cada tupla es un movimiento
    representado como (i, v, j)
    '''

    # obtiene una lista de tuplas
    # cada tupla es un movimiento
    # representado como (i, v, j)

    # ponerlo como lista de compresion
    movimientos_posibles = [(i, v, j)
                            for i, frontera_region_i in dict_fronteras_P.items()
                            for v in frontera_region_i
                            for j in dict_conjuntos_cambio_P[v]]

    # devolver todos esos movimientos
    return movimientos_posibles



# Actualizar los conjuntos de cambio y frontera para una solucion vecina


# devuelve el cojunto de cambio de todos los vertices de una particion vecina
# si se tiene el cojunto de cambio de todos los vertices de una particion
# y el movimiento que lleva de una a la otra
def obtener_funcion_H_plus_particion_vecina(grafo,
                                            P, conjuntos_cambio_P,
                                            movimiento):
    '''
    Argumentos:
        Grafo
        Particion P
        Conjuntos de cambio de la particion P
        movimiento posible a la particion P

    Devuelve
        Conjuntos de cambio de la particion P'
        donde P' es el resultado de aplicar el movimiento a P

    '''

    # tomar los componentes del movimiento (i, v, j)
    # i = movimiento[0]
    v = movimiento[1]
    # j = movimiento[2]

    # obtener el vecino en cuestion
    P_prima = hacer_movimiento(P, movimiento)

    # calcular los conjuntos de cambio del vecino en cuestion
    # inician siendo igual a las de P
    conjuntos_cambio_P_prima = conjuntos_cambio_P.copy()

    # solo se actualizan los vecinos de v y v mismo
    for u in grafo.neighbors(v):

        # actualizar el conjunto de cambio de este vecino de v
        conjuntos_cambio_P_prima[u] = funcion_H_plus(u, P_prima, grafo)

    # actualizar el conjunto de v
    conjuntos_cambio_P_prima[v] = funcion_H_plus(v, P_prima, grafo)

    return conjuntos_cambio_P_prima


# devuelve la frontera de todas las regiones de una particion vecina
# si se tiene la particion actual con todas sus fronteras,
# el movimiento que lleva de una a la otra,
# y se tienen los conjuntos de cambio del vecino (obtener_funcion_H_plus_particion_vecina)
def obtener_fronteras_particion_vecina(grafo,
                                       P, fronteras_P,
                                       movimiento,
                                       conjuntos_cambio_vecino):
    '''
    Argumentos:
        Grafo
        Particion P
        Fronteras de la particion P
        movimiento posible a la particion P
        Conjuntos de cambio de la particion P'
        donde P' es el resultado de aplicar el movimiento a P

    Devuelve
        Fronteras de la particion P'
        donde P' es el resultado de aplicar el movimiento a P

    '''

    # tomar los componentes del movimiento (i, v, j)
    i = movimiento[0]
    # v = movimiento[1]
    j = movimiento[2]

    # obtener el vecino en cuestion
    P_prima = hacer_movimiento(P, movimiento)

    # calcular las fronteras de este vecino
    # inician siendo igual que las de P
    fronteras_P_prima = fronteras_P.copy()

    # solo se actualizan las fronteras de las regiones i,j
    # se actualizan por completo
    # es decir, para cada nodo en estas regiones se checa si es frontera
    # pues segun yo todos pueden ser diferentes, con respecto a las fronteras de P
    # (Piensa en un ciclo, esto demuestra que las fronteras pueden cambiar drasticamente)

    # actualizar la frontera de i
    fronteras_P_prima[i] = funcion_frontera(
        grafo, P_prima, i, conjuntos_cambio_vecino)

    # actualizar la frontera de j
    fronteras_P_prima[j] = funcion_frontera(
        grafo, P_prima, j, conjuntos_cambio_vecino)

    return fronteras_P_prima



#  ademas de evaluar la funcion, se ve la evaluacion de cada region
def funcion_objetivo_matriz_dist_complete(grafo, P, matriz_dist):

    # hacerlo por reigones
    fo_regiones = {}

    # para cada region
    for idx_region, P_i in P.items():

        # iniciar vacio
        fo_regiones[idx_region] = 0

        # ver cuantos hay
        len_region = len(P_i)

        # tomar todos los pares de nodos de la region
        pares_nodos = list(itertools.combinations(P_i, 2))

        # por cada par de nodos
        for i, j in pares_nodos:

            # agregar su distancia normalizada
            fo_regiones[idx_region] += matriz_dist[i, j]/len_region

    return sum(fo_regiones.values()), fo_regiones



# dada una particion con su funcion objetivo y un movimiento
# calcular la funcion objetivo del vel vecino que se obtiene
# al aplicar el movimiento en la particion
# computacionalmente eficiente
def funcion_objetivo_movimiento_matriz_dist(grafo,
                                            P, f_P_regiones,
                                            movimiento,
                                            matriz_dist):

    # teniendo P con f_P y el movimiento (i, v, j)

    # tomar los elementos del movimiento
    i = movimiento[0]
    v = movimiento[1]
    j = movimiento[2]

    # solo hay que modificar para las regiones i, j

    f_P_prima_regiones = f_P_regiones.copy()

    # modificar para i, quitar lo del nodo v
    len_P_i = len(P[i])
    f_P_prima_regiones[i] *= len_P_i
    f_P_prima_regiones[i] -= sum([matriz_dist[v, u] for u in P[i]])
    f_P_prima_regiones[i] /= (len_P_i - 1)

    # modificar para j, añadir lo del nodo v
    len_P_j = len(P[j])
    f_P_prima_regiones[j] *= len_P_j
    f_P_prima_regiones[j] += sum([matriz_dist[v, u] for u in P[j]])
    f_P_prima_regiones[j] /= (len_P_j + 1)

    # devolver la cantidad deseada
    return f_P_prima_regiones, sum(f_P_prima_regiones.values())

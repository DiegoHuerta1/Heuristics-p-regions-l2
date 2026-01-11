import itertools
from typing import TypedDict
import numpy as np
import igraph

PARTITION = dict[int, list[int]]
AUX_STRUCTURE = dict[int, set[int]]
MOV = tuple[int, int, int]  # (i, v, j)
F_REGIONS = dict[int, float]

class LS_Stats(TypedDict):
    P : PARTITION
    f_P : float
    historial_f : list[float]
    time: float

# ---------------------------------------------------------------------------
# Funciones para calcular conjuntos de cambio H+(v)

def obtener_funcion_H_plus_particion_vecina(grafo: igraph.Graph, P_prima: PARTITION,
                                            conjuntos_cambio_P: AUX_STRUCTURE,
                                            movimiento: MOV) -> AUX_STRUCTURE:
    '''
    Argumentos:
        Grafo
        Particion P' (la vecina)
        Conjuntos de cambio de la particion P
        movimiento posible a la particion P
        
    Devuelve
        Conjuntos de cambio de la particion P'
        donde P' es el resultado de aplicar el movimiento a P
    '''
    # tomar los componentes del movimiento (i, v, j)
    v = movimiento[1]
    
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



def obtener_funcion_H_plus_particion(grafo: igraph.Graph, P: PARTITION) -> AUX_STRUCTURE:
    '''
    Dada la particion P
    {i: nodos_en_i}
    Calcular el conjunto de cambio de todos los vertices
    {v: H+(v)}
    '''
    conjuntos_cambio = {node_v: funcion_H_plus(node_v, P, grafo)
                        for node_v in range(grafo.vcount())}
    return conjuntos_cambio

def funcion_H_plus(v: int, P: PARTITION, grafo: igraph.Graph) -> set[int]:
    # tomar H(v)
    H_v = funcion_H(v, P, grafo)
    # tomar h(v)
    h_v = funcion_h(v, P)
    # devolver la diferencia
    return H_v - {h_v}

def funcion_H(v: int, P: PARTITION, grafo: igraph.Graph) -> set[int]:
    # tomar las etiquetas a devolver
    etiquetas_devolver = [funcion_h(u, P)
                          for u in grafo.neighbors(v)]
    # devolver como conjunto
    return set(etiquetas_devolver)

def funcion_h(v: int, P: PARTITION) -> int:
    # iterar en las regiones con su indice
    for idx_region, region in P.items():
        # si el nodo esta en esta region
        if v in region:
            # devolver el indice
            return idx_region
    raise Exception(f"Nodo {v} no se encuentra en la particion")


# ---------------------------------------------------------------------------
# Funciones para calcular fronteras de una particion

def obtener_fronteras_particion_vecina(grafo: igraph.Graph, P_prima: PARTITION, 
                                       fronteras_P: AUX_STRUCTURE,
                                       movimiento: MOV,
                                       conjuntos_cambio_vecino: AUX_STRUCTURE) -> AUX_STRUCTURE:
    '''
    Argumentos:
        Grafo
        Particion P' (la vecina)
        Fronteras de la particion P
        movimiento posible a la particion P
        Conjuntos de cambio de la particion P'
        
    Devuelve
        Fronteras de la particion P'
        donde P' es el resultado de aplicar el movimiento a P
    '''
    # tomar los componentes del movimiento (i, v, j)
    i = movimiento[0]
    j = movimiento[2]

    # calcular las fronteras de este vecino
    # inician siendo igual que las de P
    fronteras_P_prima = fronteras_P.copy()
    
    # solo se actualizan las fronteras de las regiones i,j
    # se actualizan por completo
    # es decir, para cada nodo en estas regiones se checa si es frontera
    # pues segun yo todos pueden ser diferentes, con respecto a las fronteras de P
    # (Piensa en un ciclo, esto demuestra que las fronteras pueden cambiar drasticamente)
    
    # actualizar la frontera de i
    fronteras_P_prima[i] = funcion_frontera(grafo, P_prima, i, conjuntos_cambio_vecino)
    # actualizar la frontera de j
    fronteras_P_prima[j] = funcion_frontera(grafo, P_prima, j, conjuntos_cambio_vecino)
    return fronteras_P_prima

def obtener_fronteras_particion(grafo: igraph.Graph, P: PARTITION,
                                conjuntos_cambio_vertices: AUX_STRUCTURE) -> AUX_STRUCTURE:
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

def funcion_frontera(grafo: igraph.Graph, P: PARTITION, i: int,
                     conjuntos_cambio_vertices: AUX_STRUCTURE) -> set[int]:
    # primero filtrar los que tengan conjunto de cambio no vacio
    frontera_preliminar = [v for v in P[i]
                           if len(conjuntos_cambio_vertices[v]) > 0]
    # solo conservar los que se puedan quitar de esa ragion
    frontera = [v for v in frontera_preliminar
                if se_puede_quitar_nodo_region(grafo, P, i, v)]
    # devolver la frontera
    return set(frontera)


def se_puede_quitar_nodo_region(grafo: igraph.Graph, P: PARTITION,
                                i: int, v: int) -> bool:
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
    v_subgrafo = subgrafo_Gi.vs.find(name= nombre_v).index
    
    # quitara v de la subgrafica
    subgrafo_Gi.delete_vertices(v_subgrafo)
    
    # ver si es que se rompio factibilidad
    if subgrafo_Gi.vcount() == 0 or not subgrafo_Gi.is_connected():
        
        # se vuelve disconexo o vacio, no se puede quitar
        return False
    
    # si no se devuelve False pues se devueelve True
    # v si se puede quitar de la subgrafica
    return True

# ---------------------------------------------------------------------------
# Movimientos 

def hacer_movimiento(P: PARTITION, mov: MOV) -> PARTITION:
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


def get_posibles_movimientos(dict_fronteras_P: AUX_STRUCTURE,
                             dict_conjuntos_cambio_P: AUX_STRUCTURE) -> list[MOV]:
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
    movimientos_posibles = [(i, v, j)
                            for i, frontera_region_i in dict_fronteras_P.items()
                            for v in frontera_region_i
                            for j in dict_conjuntos_cambio_P[v]]
    # devolver todos esos movimientos
    return movimientos_posibles

# ---------------------------------------------------------------------------
# Evaluacion de la funcion objetivo

def funcion_objetivo_matriz_dist_complete(P: PARTITION,
                                         diss_matrix: np.ndarray) -> tuple[float, F_REGIONS]:
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
            fo_regiones[idx_region] += diss_matrix[i, j]/len_region

    return sum(fo_regiones.values()), fo_regiones


def funcion_objetivo_movimiento_matriz_dist(P: PARTITION, f_P_regiones: F_REGIONS,
                                            movimiento: MOV,
                                            diss_matrix: np.ndarray) -> tuple[F_REGIONS, float]:
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
    f_P_prima_regiones[i] -= sum([diss_matrix[v, u] for u in P[i]])
    f_P_prima_regiones[i] /= (len_P_i - 1)

    # modificar para j, añadir lo del nodo v
    len_P_j = len(P[j])
    f_P_prima_regiones[j] *= len_P_j
    f_P_prima_regiones[j] += sum([diss_matrix[v, u] for u in P[j]])
    f_P_prima_regiones[j] /= (len_P_j + 1)

    # devolver la cantidad deseada
    return f_P_prima_regiones, sum(f_P_prima_regiones.values())

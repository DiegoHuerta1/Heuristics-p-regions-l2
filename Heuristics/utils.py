import numpy as np
import igraph 
import itertools
import geopandas as gpd
import pandas as pd
from shapely.geometry import Polygon, MultiPolygon



def generate_dissimilarity_matrix(graph: igraph.Graph) -> np.ndarray:
    """
    Generate a dissimilarity matrix

    A_{i, j} = ||x_i - x_j||^2 

    where || . || is the euclidean norm
    and x_i is the vector of attributes of node i.
    """

    # Initialize with zeros
    size = graph.vcount()
    dissimilarity_matrix = np.zeros((size, size))

    # iterate over all pairs
    for i in range(size):
        x_i = graph.vs[i]["x"]
        for j in range(i + 1, size):
            x_j = graph.vs[j]["x"]

            # compute the dissimilarity 
            dissimilarity_matrix[i, j] = np.linalg.norm(x_i - x_j) ** 2
            dissimilarity_matrix[j, i] = dissimilarity_matrix[i, j]

    return dissimilarity_matrix


def l2_objective_function_diss_matrix(P: dict[int, list[int]], diss_matrix: np.ndarray) -> float:
    """
    Compute the l2 objective function for a given partition P
    using the pre computed dissimilarity matrix

    Args:
        graph (igraph.Graph): Graph instance
        P (dict[int, list[int]]): Partition of the graph
        diss_matrix (np.ndarray): Dissimilarity matrix

    Returns:
        float: Evaluation of the partition P
    """

    # start sum 
    sum = 0.0 

    # for each region
    for P_k in P.values():
        n_k = len(P_k)
        # for each pair of nodes in the region
        node_pairs = itertools.combinations(P_k, 2)
        for i, j in node_pairs:

            # add the corresponding term
            sum += diss_matrix[i, j]/n_k

    return float(sum)


def get_node_region_idx(P: dict[int, list], v) -> None | int:
    """ 
    Returns the index of the region of node v, in partition P
    """

    # Look in all regions
    for k, P_k in P.items():
        if v in P_k:
            return k
        
    # It was not found
    return None


def compute_P_names(graph: igraph.Graph, P: dict[int, list[int]]) -> dict[int, list[str]]:
    """
    Given a partition using node indices (int) transform it to consider node names

    Args:
        graph (igraph.Graph): Graph with name attributes
        P (dict[int, list[int]]): Partition of G, with node indices

    Returns:
        dict[int, list[str]]: Parition of G, with node names
    """
    # Use the node attributes to change from idx to name
    P_names = {}
    for k, P_k in P.items():
        P_names[k] = [graph.vs[v]["name"] for v in P_k]
    return P_names


def labels_to_P(labels: tuple[int], K: int) -> dict[int, list[int]]:
    """
    Transform cluster labels to a dict partition

    Args:
        labels (tuple[int]): Cluster label for each node. Domain: {1, 2, ..., K}
        K (int): Number of regions.

    Returns:
        dict[int, list[int]]: Dictionary of the corresponding partition P = {k: P_k}
    """
    labels_arr = np.array(labels)
    
    # For each region, construct P_k in the dict
    P: dict[int, list[int]] = {}
    for k in range(1, K+1):
        if k in np.unique(labels_arr):
            # Add node indices to region
            P_k = np.where(labels_arr == k)[0]
            P[k] = list(P_k)
        else:
            # Empty region
            P[k] = []
            
    return P


def igraph_to_gdf(graph: igraph.Graph) -> gpd.GeoDataFrame:
    """
    Convert an igraph.Graph to a GeoDataFrame
    such that it has the same node attributes as columns,
    and the queen weigths of the gdf are exactly the edges of the graph

    Args:
        graph (igraph.Graph): Graph with attributes "x" for each node

    Returns:
        gdp.GeoDataFrame: GeoDataFrame with the same node attributes as columns x_i
    """

    # Construct a DataFrame row by row (a node is a row)
    rows = []
    for v in graph.vs:
        # name
        dict_name = {"name": v["name"]}
        # attributes
        dict_attributes = {
            f"x_{i+1}": v["x"][i] for i in range(len(v["x"]))
        }
        rows.append(dict_name | dict_attributes)
    # Make a df with this information
    df = pd.DataFrame(rows)

    # Obtain node geometries to make a GeoDataFrame
    geometries = obtain_node_geometries(graph)
    gdf = gpd.GeoDataFrame(df, geometry = geometries)

    return gdf


def obtain_node_geometries(graph: igraph.Graph) -> list[MultiPolygon]:
    """
    Create geometry for each node of a graph,
    such that the queen weigths of the geometries are exactly the edges of the graph

    Args:
        graph (igraph.Graph): Graph instance

    Returns:
        list[MultiPolygon]: List of MultiPolygon, one for each node
    """

    # Create a list of Polygon for each node
    lists_polygons: dict[int, list] = {v.index: [] for v in graph.vs}
    for u, v in graph.get_edgelist():
        # Construct the Polygon associated with this edge, add it to the nodes
        edge_polygon = Polygon([(u, v)] * 4)
        lists_polygons[u].append(edge_polygon)
        lists_polygons[v].append(edge_polygon)

    # Construct the MultiPolygons
    geometries = [MultiPolygon(lists_polygons[v.index]) for v in graph.vs]
    return geometries


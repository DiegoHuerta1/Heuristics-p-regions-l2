import matplotlib.colors 
import seaborn as sns
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.lines

from .utils import get_node_region_idx



def get_dict_palette(num_colors: int, palette_name: str) -> dict:
    """ 
    Get a dictionaty of a color palette

    Palette name options:
    Deep, muted, bright, pastel, dark, colorblind, husl, hls
    """

    # Start with seaborn palette
    colors = sns.color_palette(palette_name, num_colors)
    # Transforms to hex, and dict
    hex_colors = [matplotlib.colors.to_hex(rgb) for rgb in colors]
    dict_palette = {(idx+1): color for idx, color in enumerate(hex_colors)}

    return dict_palette


def draw_partition_map(gdf_file: gpd.GeoDataFrame, P_names: dict,
                       palette_name: str = "husl", figsize: tuple = (4, 4),
                       title: str | None = None):
    """ 
    Draws a partition P_names (that has names of nodes, i.e CVEGEO)
    considering the polygons in gdf_file
    """

    # Get the color palette of the partition
    colors_P: dict = get_dict_palette(len(P_names), palette_name)
    # For each unit, get the color based on the partition
    units_colors = gdf_file['CVEGEO'].apply(lambda n: colors_P[get_node_region_idx(P_names, n)])

    # Make the figure
    fig, ax = plt.subplots(figsize = figsize)
    gdf_file.plot(color = units_colors, linewidth=0.8, ax=ax, edgecolor='0.8')
    # legend
    legend_hanldes = []
    for k, color_k in colors_P.items():
        legend_hanldes.append(matplotlib.lines.Line2D([0], [0],  marker='o', color='w', 
                                                     label = f"P_{k}", markerfacecolor = color_k,
                                                     markersize=10))            
    ax.legend(handles= legend_hanldes, loc='upper right', bbox_to_anchor=(1.3, 0.9))
    # final details
    if title is not None:
        ax.set_title(title)
    ax.axis('off')
    plt.show()
    plt.close()




################### CMCM 2018 ###################
# Author: Garrett Tetrault
# Create isochrones of Ithaca, NY for graphical
# representation of travel times from a specified
# location.  Code based on example from:
# Github User: gboeing
# File Location: osmnx-examples/notebooks/
#                13-isolines-isochrones.ipynb

import osmnx as ox
import networkx as nx
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
from shapely.geometry import Point, LineString, Polygon
from descartes import PolygonPatch

################ Helper Functions ###############
# Create the polygons of the isochrone to plot. Returns
# array of arrays of polygons, with one array corresponding
# to each center.
def make_iso_polys(G, centers, trip_times):
    # Define the width of the polygons around
    #  edges and nodes.
    edge_buff = 40
    node_buff = 25

    # Init array to store polygons.
    isochrone_polys = [[None]] * len(centers)
    
    # For each center, create the corresponding 
    # polygon for the specified travel time.
    for i in range(0, len(centers)):

        isochrone_polys[i] = []
        for trip_time in sorted(trip_times, reverse=True):

            # Generate the subgraph.
            subgraph = nx.ego_graph(G, centers[i], 
                radius=trip_time, distance='time')

            # Format the data from the subgraph into 
            # data frames.
            node_points = [Point((data['x'], data['y'])) for
                node, data in subgraph.nodes(data=True)]
            nodes_gdf = gpd.GeoDataFrame({'id': subgraph.nodes()}, 
                geometry=node_points)
            nodes_gdf = nodes_gdf.set_index('id')

            # Get edges that the isochrone encapsulates.
            edge_lines = []
            for n_fr, n_to in subgraph.edges():
                f = nodes_gdf.loc[n_fr].geometry
                t = nodes_gdf.loc[n_to].geometry
                edge_lines.append(LineString([f,t]))

            # Construct the poygon that represents the 
            # isochrone.
            n = nodes_gdf.buffer(node_buff).geometry
            e = gpd.GeoSeries(edge_lines).buffer(edge_buff).geometry
            all_gs = list(n) + list(e)
            new_iso = Polygon(gpd.GeoSeries(all_gs).unary_union)

            # Try to fill in surrounded areas so shapes 
            # will appear solid and blocks without white 
            # space inside them.
            new_iso = Polygon(new_iso.exterior)
            isochrone_polys[i].append(new_iso)
            
    return isochrone_polys

################# Initial Setup #################
ox.config(log_console=True, use_cache=True)
title = 'Isochrone'

# Configure the place, network type, trip times, and travel speed.
place = 'Ithaca, NY, USA'

# Starting locations of ambulances/
centers = [(42.45444, -76.51536), (42.45233, -76.49427), 
    (42.43267, -76.48417), (42.43917, -76.50247), (42.43934, -76.51246)]

network_type = 'drive'
trip_times = [3,4,5,6,7,8,9,10] # In minutes.
travel_speed = 10 # In mph.

################ Graph Processing ###############

# Download the street network.
G = ox.graph_from_place(place, network_type=network_type)

# Find nearest locations on graph to centers 
# defined above.
for i in range(0, len(centers)): 
    centers[i] = ox.get_nearest_node(G, centers[i])

G = ox.project_graph(G)

# Miles per hour to meter per minute.
meters_per_minute = (travel_speed * 5280 * 12 * 2.54) / (100 * 60)

# Add an edge attribute for time in minutes 
# required to traverse each edge.
for u, v, k, data in G.edges(data=True, keys=True):
    data['time'] = data['length'] / meters_per_minute

################ Plotting Results ###############

# Get one color for each isochrone polygon level.
# Note that we split the color map about 6 to 
# highlight what levels are within our tolerance.
six_min_loc = trip_times.index(6) + 1

iso_colors = ox.get_colors(n=len(trip_times)-six_min_loc, 
    cmap='Oranges', start=0.3, stop=0.7, return_hex=True)[::-1]

iso_colors += ox.get_colors(n=six_min_loc, 
    cmap='Greens', start=0.3, stop=1.0, return_hex=True)

# Only show the nodes corresponding to the 
# centers as black.
node_color = ['k' if node in centers else 'none' 
    for node in G.nodes()]
node_size = [30 if node in centers else 0 
    for node in G.nodes()]

# Create figure from graph.
fig, ax = ox.plot_graph(G, fig_height=6, show=False, 
    close=False, edge_color='k', edge_alpha=0.3, 
    node_color=node_color, node_size=node_size)

isochrone_polys = make_iso_polys(G, centers, trip_times)

# Plot each poly for each isochrone beginning at the 
# lowest level (the longest travel time).
for i in range(0, len(trip_times)):
    for iso_polys in isochrone_polys:
        # Add polygon to plot.
        patch = PolygonPatch(iso_polys[i], fc=iso_colors[i], 
            ec='none', alpha=0.85, zorder=-1)
        ax.add_patch(patch)

# Create custom label with trip lengths 
# corresponding to colors.
time_labels = [str(trip_time) + ' min' 
    for trip_time in trip_times]
handles = [mpatches.Patch(color=color, label=label) 
    for color, label in zip(iso_colors[::-1], time_labels)]

# Add marker for starting location to legend.
handles = [mlines.Line2D([],[], marker='.', markersize=15,
    linewidth=0, color='k', label='Starting Location')] + handles

ax.legend(handles=handles, loc='center right', fontsize='small')

# Display final result.
# fig.suptitle(title, fontsize=12)
plt.show()

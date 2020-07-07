################### CMCM 2018 ###################
# Author: Garrett Tetrault
# Find the average travel time from the given
#  centers to every point in Ithaca.

import itertools as it
import osmnx as ox
import networkx as nx
import geopandas as gpd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
from shapely.geometry import Point, LineString, Polygon
from descartes import PolygonPatch

################# Initial Setup #################
ox.config(log_console=True, use_cache=True)

# Configure the place, network type, trip times, and travel speed.
place = 'Ithaca, NY, USA'
centers = [(42.45444, -76.51536)]

network_type = 'drive'
trip_times = np.linspace(3,30,num=27) # In minutes.
travel_speed = 10 # In mph.

################ Graph Processing ###############

# Download the street network.
G = ox.graph_from_place(place, network_type=network_type)

# Find nearest locations on graph to centers 
# defined above.
center_nodes = [0] * len(centers)
for i in range(0, len(centers)): 
    center_nodes[i] = ox.get_nearest_node(G, centers[i])

G = ox.project_graph(G)

# Miles per hour to meter per minute.
meters_per_minute = (travel_speed * 5280 * 12 * 2.54) / (100 * 60)

# Add an edge attribute for time in minutes required to 
# traverse each edge.
for u, v, k, data in G.edges(data=True, keys=True):
    data['time'] = data['length'] / meters_per_minute

################## Find Average #################
sub_graphs = [None] * len(trip_times)

# For every trip time, get the total coverage for that 
# trip time amongst all centers.
for i in range(0, len(trip_times)):
    sub_graphs[i] = nx.Graph()

    # Get all nodes from one time from all centers.
    for center_node in center_nodes:
        sub_graphs[i].add_nodes_from(nx.ego_graph(G, center_node, 
            radius=trip_times[i], distance='time'))

    # Remove nodes that have been counted previously.
    for j in range(i-1, -1, -1):
        sub_graphs[i].remove_nodes_from(sub_graphs[j])

# Compute the weighted average to find the average 
# time of travel.
average = 0
for sub, time in zip(sub_graphs, trip_times):
    average += sub.number_of_nodes() * time
average /= G.number_of_nodes()

print('Average = ' + str(average))
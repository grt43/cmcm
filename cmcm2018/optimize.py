################### CMCM 2018 ###################
# Author: Garrett Tetrault
# Create from a specified list of locations,
# find the combinations that cover the area
# of Ithaca within some given timeframe.

import itertools as it
import osmnx as ox
import networkx as nx
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
from shapely.geometry import Point, LineString, Polygon
from descartes import PolygonPatch

################# Initial Setup #################
ox.config(log_console=True, use_cache=True)

# Configure the place, network type, trip times, and travel speed.
place = 'Ithaca, NY, USA'
centers = [(42.46089, -76.50496), (42.45444, -76.51536),
    (42.45233, -76.49427), (42.44400, 76.47969),
    (42.43267, -76.48417), (42.43917, -76.50247),
    (42.43934, -76.51246), (42.44795, -76.51619),
    (42.43713, -76.51752)]

# [(42.440853, -76.509350), (42.461354, -76.492477), 
#     (42.44034,-76.493039), (42.442098,-76.524386)]
network_type = 'drive'
trip_time = 6 # In minutes.
travel_speed = 10 # In mph.
num_centers = 6

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

#################### Optimize ###################

# Zip centers and center_nodes to preserve relation 
# in location.
combinations = list(
    it.combinations(zip(centers, center_nodes), num_centers))

# Init list for sizes of the subgraphs.
sub_sizes = [0] * len(combinations)

# For each combination, calculate the size of the coverage 
# by finding how many nodes the union of all subgraphs 
# reach within the specified time.
for i in range(0, len(combinations)):
    subgraph = nx.Graph()
    for center, center_node in combinations[i]:

        # Add all nodes of subgraph from the current 
        # center to the total converage, avoiding 
        # repetitions.
        subgraph.add_nodes_from(nx.ego_graph(G, center_node, 
            radius=trip_time, distance='time').nodes())
    
    sub_sizes[i] = subgraph.number_of_nodes()

# Find the maximum coverage and get corresponding 
# combination.
max_size = max(sub_sizes)
max_combo = combinations[sub_sizes.index(max_size)]

# Output data.
print(max_combo)
print('Percent Coverage = ' + 
    str(max_size / G.number_of_nodes()))


# This performs core-periphery analyses on three artificial data sets and one real world data set

# The artificial data sets are simple graphs which collectively motivate the generalizations underlying cpip
# by showing how the different parameters affect core assignment -- these networks are UNDIRECTED / BI-DIRECTIONAL!

# The real world data set is data on shipments between different metro areas via railway -- this netowrk is DIRECTED!

# Importing required modules

import pandas as pd
import numpy as np
import networkx as nx
import geopandas as gpd
from matplotlib import pyplot as plt
from cpip import *

# Declaring root of file paths -- set this up for wherever you want to store data files and write outputs

root = 'C:/Users/User/Documents/Data/cpip_test/'

# Loading the data -- shipping data from BTS
# --->>> THIS CAN BE DOWNLOADED FROM: https://www.census.gov/data/datasets/2012/econ/cfs/2012-pums-files.html

data = pd.read_csv(root + 'cfs_2012_pumf_csv.txt')

# Create a new data frame with only shipments between different, known CFS_AREAs that is transported via railway

data = data[data['ORIG_CFS_AREA'] != data['DEST_CFS_AREA']]
data = data[data['ORIG_MA'] != 00000]
data = data[data['DEST_MA'] != 00000]
data = data[data['MODE'] == 6].reset_index(drop = True)

# Re-label the CFS_AREA codes to make useful indices

idx_tab = {'36-104':0, '13-122':1, '01-142':2, '25-148':3, '33-148':4, '44-148':5, '36-160':6,
           '37-172':7, '17-176':8, '18-176':9, '21-178':10, '39-178':11, '39-184':12, '39-198':13,
           '48-204':14, '48-206':15, '39-212':16, '08-216':17, '26-220':18, '48-238':19, '18-258':20,
           '06-260':21, '26-266':22, '37-268':23, '45-273':24, '48-288':25, '18-294':26, '12-300':27,
           '20-312':28, '29-312':29, '47-314':30, '32-332':31, '06-348':32, '21-350':33, '47-368':34,
           '12-370':35, '55-376':36, '27-378':37, '01-380':38, '47-400':39, '22-406':40, '09-408':41,
           '34-408':42, '36-408':43, '42-408':44, '40-416':45, '31-420':46, '12-422':47, '10-428':48,
           '34-428':49, '42-428':50, '42-430':51, '41-440':52, '53-440':53, '37-450':54, '36-464':55,
           '06-472':56, '17-476':57, '29-476':58, '49-482':59, '06-488':60, '13-496':61, '53-500':62,
           '04-536':63, '40-538':64, '51-545':65, '20-556':66, '48-12420':67, '24-12580':68, '22-12940':69,
           '48-13140':70, '45-16700':71, '09-25540':72, '22-29340':73, '48-29700':74, '04-38060':75,
           '51-40060':76, '48-41700':77, '06-41740':78, '12-45300':79, '15-46520':80, '11-47900':81,
           '24-47900':82, '51-47900':83, '01-99999':84, '04-99999':85, '06-99999':86, '08-99999':87,
           '09-99999':88, '10-99999':89, '12-99999':90, '13-99999':91, '15-99999':92, '17-99999':93,
           '18-99999':94, '20-99999':95, '21-99999':96, '22-99999':97, '24-99999':98, '25-99999':99,
           '26-99999':100, '27-99999':101, '29-99999':102, '31-99999':103, '32-99999':104, '33-99999':105,
           '36-99999':106, '37-99999':107, '39-99999':108, '40-99999':109, '41-99999':110, '42-99999':111,
           '45-99999':112, '47-99999':113, '48-99999':114, '49-99999':115, '51-99999':116, '53-99999':117,
           '55-99999':118, '02-99999':119, '05-99999':120, '16-99999':121, '19-99999':122, '23-99999':123,
           '28-99999':124, '30-99999':125, '35-99999':126, '38-99999':127, '46-99999':128, '50-99999':129,
           '54-99999':130, '56-99999':131}

# Create data frames containing total shipments between states by NAICS category

freqs = np.zeros((132,132))
values = np.zeros((132,132))

for row in range(len(data)):
    
    print(str(row+1) + ' of ' + str(len(data)))
    freqs[idx_tab[data['ORIG_CFS_AREA'][row]]][idx_tab[data['DEST_CFS_AREA'][row]]] += 1
    values[idx_tab[data['ORIG_CFS_AREA'][row]]][idx_tab[data['DEST_CFS_AREA'][row]]] += data['SHIPMT_VALUE'][row]

freqs = pd.DataFrame(freqs, columns = list(idx_tab.keys()))
values = pd.DataFrame(values, columns = list(idx_tab.keys()))

# Write outputs to file (source/supply data)

freqs.to_csv(root + 'frequencies.csv', index = False)
values.to_csv(root + 'values.csv', index = False)

# Create sink data (ship to data rather than ship from/source data == demand data) by transposing and saving

freqs_t = pd.DataFrame(freqs.values.T, columns = freqs.columns)
values_t = pd.DataFrame(values.values.T, columns = values.columns)

freqs_t.to_csv(root + 'frequencies_t.csv', index = False)
values_t.to_csv(root + 'values_t.csv', index = False)

# Running cpip to perform the core-periphery analysis on the railway transportation data

core_freqs_sources = cpip(root + 'frequencies.csv', 1.2, 1)
core_vals_sources = cpip(root + 'values.csv', 1.2, 1)
core_freqs_sinks = cpip(root + 'frequencies_t.csv', 1.2, 1)
core_vals_sinks = cpip(root + 'values_t.csv', 1.2, 1)

# Running cpip to perform the core-periphery analyses for the artifical data sets

# Showing that the idealized core structure exists on (a) but not (b) or (c)

core_a_1 = cpip(root + 'a.csv', 1, 1)
core_b_1 = cpip(root + 'b.csv', 1, 1)
core_c_1 = cpip(root + 'c.csv', 1, 1)

# Core-periphery analysis on (b) and (c) with restriction that psi = 1 (i.e., the core must be a clique)

core_b_2 = cpip(root + 'b.csv', 1.5, 1)
core_c_2 = cpip(root + 'c.csv', 1.5, 1)

# Core-periphery analysis on (c) with relaxed constraint on core density (psi = 1.2)

core_c_3 = cpip(root + 'c.csv', 1.5, 1.2)

# Writing results of the core-periphery analyses to file

all_cores_vals = [core_freqs_sources, core_vals_sources, core_freqs_sinks, core_vals_sinks,
                   core_a_1, core_b_1, core_c_1, core_b_2, core_c_2, core_c_3]
all_cores_names = ['Railway - Sources - Frequencies', 'Railway - Sources - Values',
                   'Railway - Sinks - Frequencies', 'Railway - Sinks - Values', 'Graph A (1,1)',
                   'Graph B (1,1)', 'Graph C (1,1)', 'Graph B (1.5,1)', 'Graph C (1.5,1)', 'Graph C (1.5,1.2)']

for c in all_cores_vals:
    
    for i in range(len(c)):
        
        c[i] = c[i].replace('_','-')

with open(root + 'cpip_examples_results.txt', 'w') as file:
    
    for i in range(len(all_cores_vals)):
        
        file.write(str(all_cores_names[i]) + ':\n')
        file.write(str(len(all_cores_vals[i])) + ' members in the core\n')
        file.write(str(all_cores_vals[i]) + '\n\n')

file.close()

# Create network visualizations using networkx and geopandas

# Read in the shapefile for US states and drop non-contiguous states

shpdata = gpd.read_file(root + 'cb_2018_us_state_500k.shp')
drops = ['Commonwealth of the Northern Mariana Islands', 'Guam', 'Hawaii',
         'United States Virgin Islands', 'American Samoa', 'Puerto Rico', 'Alaska']
shpdata = shpdata[~shpdata['NAME'].isin(drops)]

# Further data prep for plots

f_sources = np.zeros((129,129))
f_sinks = np.zeros((129,129))

for cfs_code in ['02-99999', '15-46520', '15-99999']:
    
    freqs = freqs.drop(cfs_code, axis = 1).drop(list(idx_tab.keys()).index(cfs_code), axis = 0).reset_index(drop = True)
    freqs_t = freqs_t.drop(cfs_code, axis = 1).drop(list(idx_tab.keys()).index(cfs_code), axis = 0).reset_index(drop = True)

for i in range(len(f_sources)):
    
    for j in range(len(f_sources)):
        
        if freqs[freqs.columns[i]][j] > 0:
            
            f_sources[i][j] = 1
        
        if freqs_t[freqs_t.columns[i]][j] > 0:
            
            f_sinks[i][j] = 1
            
# Defining longitude and lattitude coordinates for vertices to be placed on the maps

pos = [[-73.87,42.76],[-83.92,33.91],[-86.7,33.45],[-71.13,42.35],[-71.47,42.83],[-71.62,41.85],[-78.7,42.86],
       [-80.86,35.3],[-87.94,41.83],[-86.96,41.46],[-84.54,38.54],[-84.27,39.39],[-81.69,41.37],[-83.01,40.07],
       [-97.78,27.5],[-97.14,32.99],[-84.31,39.87],[-104.87,39.77],[-83.26,42.41],[-106.06,31.78],[-85.24,41.08],
       [-119.71,36.76],[-85.66,42.95],[-80.08,36.05],[-82.11,34.89],[-95.46,29.49],[-86.18,39.61],[-81.5,30.29],
       [-95.19,39.12],[-94.2,38.71],[-84.1,36.1],[-115.15,36.53],[-118.21,34.26],[-85.82,37.93],[-89.83,35.42],
       [-80.33,25.69],[-88.15,43.16],[-93.47,44.95],[-87.94,30.81],[-86.75,36.41],[-90.03,29.8],[-73.34,41.34],
       [-74.31,40.75],[-74.02,41.39],[-75.58,41.24],[-97.51,35.63],[-96.37,41.29],[-81.54,28.48],[-75.7,39.48],
       [-74.9,39.69],[-75.47,40.17],[-80,40.54],[-122.9,45.46],[-121.97,45.92],[-78.71,35.72],[-77.52,43.19],
       [-121.49,38.76],[-89.7,38.68],[-90.55,38.63],[-111.82,40.67],[-122.02,37.72],[-81.15,31.99],[-122.04,47.79],
       [-110.94,32.19],[-96.16,36.39],[-76.44,36.68],[-97.52,37.61],[-97.89,30.42],[-76.68,39.3],[-91.36,30.56],
       [-94.18,30.43],[-80.08,32.94],[-72.67,41.86],[-93.48,30.35],[-99.7,28.01],[-112.1,33.57],[-77.45,37.43],
       [-99.14,29.65],[-117.05,32.7],[-82.66,28.07],[-77.05,38.93],[-77.29,39.17],[-77.85,38.95],[-87.41,32.29],
       [-112.26,35.4],[-117.57,35.79],[-106.04,37.78],[-72.26,41.46],[-75.4,38.69],[-84.53,30.33],[-84.52,32.23],
       [-90.58,40.49],[-86.57,38.52],[-100.67,38.66],[-82.99,37.58],[-92.85,32.4],[-77.54,39.63],[-72.55,42.41],
       [-84.67,43.94],[-95.75,44.23],[-92.47,37.35],[-100.38,41.67],[-116.74,40.45],[-71.6,43.68],[-74.68,43.57],
       [-78.1,34.85],[-83.71,41.01],[-95.67,34.51],[-120,43.54],[-77.26,40.84],[-80.9,33.86],[-86.2,35.12],
       [-101.33,32.56],[-111.79,38.3],[-79.33,37.24],[-119.35,46.93],[-89.77,44.87],[-92.28,34.87],[-114.62,43.25],
       [-93.93,42.3],[-69.21,45.52],[-89.73,32.79],[-109.91,46.57],[-106.43,34.96],[-100.55,47.39],[-100.4,44.2],
       [-72.95,43.79],[-81,38.53],[-107.86,42.88]]

# Creating sources plot

cm = plt.get_cmap('tab20')
plt.figure()
base = shpdata.plot(color = 'lightgrey', edgecolor = 'black')
node_sizes = [50 if idx in core_freqs_sources else 5 for idx in list(freqs.columns)]
node_colors = [cm(1/3) if size == 50 else cm(0) for size in node_sizes]
g = nx.Graph(f_sources)
nx.draw(g, pos, node_size = node_sizes, node_color = node_colors, edge_color = 'lightgrey')
plt.title('Railway Shipments by Source', loc = 'center', fontsize = 20, color = 'black')
plt.savefig(root + 'map_sources.eps')
plt.show()

# Creating sinks plot

cm = plt.get_cmap('tab20')
plt.figure()
base = shpdata.plot(color = 'lightgrey', edgecolor = 'black')
node_sizes = [50 if idx in core_freqs_sinks else 5 for idx in list(freqs_t.columns)]
node_colors = [cm(1/3) if size == 50 else cm(0) for size in node_sizes]
g = nx.Graph(f_sinks)
nx.draw(g, pos, node_size = node_sizes, node_color = node_colors, edge_color = 'lightgrey')
plt.title('Railway Shipments by Destination', loc = 'center', fontsize = 20, color = 'black')
plt.savefig(root + 'map_sinks.eps')
plt.show()

# Repeat without edges

# Creating sources plot

cm = plt.get_cmap('tab20')
plt.figure()
base = shpdata.plot(color = 'lightgrey', edgecolor = 'black')
node_sizes = [50 if idx in core_freqs_sources else 5 for idx in list(freqs.columns)]
node_colors = [cm(1/3) if size == 50 else cm(0) for size in node_sizes]
g = nx.Graph(f_sources)
nx.draw_networkx_nodes(g, pos, node_size = node_sizes, node_color = node_colors)
plt.axis('off')
plt.title('Railway Shipments by Source', loc = 'center', fontsize = 20, color = 'black')
plt.savefig(root + 'map_sources_no_edges.eps')
plt.show()

# Creating sinks plot

cm = plt.get_cmap('tab20')
plt.figure()
base = shpdata.plot(color = 'lightgrey', edgecolor = 'black')
node_sizes = [50 if idx in core_freqs_sinks else 5 for idx in list(freqs_t.columns)]
node_colors = [cm(1/3) if size == 50 else cm(0) for size in node_sizes]
g = nx.Graph(f_sinks)
nx.draw_networkx_nodes(g, pos, node_size = node_sizes, node_color = node_colors)
plt.axis('off')
plt.title('Railway Shipments by Destination', loc = 'center', fontsize = 20, color = 'black')
plt.savefig(root + 'map_sinks_no_edges.eps')
plt.show()


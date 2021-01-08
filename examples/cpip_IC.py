# This script performs the core-periphery analysis of international collaboration
# using the MDPI data from Cary & Rockwell, 2020 for paper:
# Core-Periphery Models via Integer Programming: Maximizing the Influence of the Core

# Importing required modules

import pandas as pd
import geopandas as gpd
from matplotlib import pyplot as plt
import networkx as nx
from cpip import *

# Runnin the analysis

# Locating the data for years 2010 - 2019

data_path = 'C:/Users/User/Documents/Data/MDPI/'
files = ['M_' + str(i) + '.csv' for i in range(2010,2020)]

# Main loop

for f in files:
    
    core0 = cpip(data_path + f, 1, 1)
    core1 = cpip(data_path + f, 1.1, 1)
    core2 = cpip(data_path + f, 1.2, 1)
    core3 = cpip(data_path + f, 1.3, 1)
    core4 = cpip(data_path + f, 1.4, 1)
    print(f[2:6])
    print(core0)
    print(core1)
    print(core2)
    print(core3)
    print(core4)

# Plotting Southern African part of this network on a world political map for year 2019

cm = plt.get_cmap('tab20')
plt.figure()
data = pd.read_csv(data_path + files[-1])
sa = ['Angola', 'Zambia', 'Malawi', 'Mozambique', 'Namibia', 'Botswana',
      'Zimbabwe', 'South Africa', 'eSwatini', 'Lesotho']
world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
world = world[world.name.isin(sa)]
base = world.plot(color = 'white', edgecolor = 'black')
sa = list(sorted(['Eswatini' if s == 'eSwatini' else s for s in sa]))
idxset = [list(data.columns).index(s) for s in sa]
data = data[sa]
data = data[data.index.isin(idxset)]
rwidths = [data.values[i][j] for i in range(len(data)) for j in range(i,len(data))]
rwidths = [5 * width / max(rwidths) for width in rwidths]
rnodes = [sum(list(data[c])) for c in data.columns]
rg = nx.MultiGraph([(i,j) for i in range(len(data)) for j in range(i,len(data))])
latlon = [[-12.4,17.4], [-23.9,22.7], [-26.4,31.6], [-29.6,28.1], [-13,33.8],
          [-18,35], [-21,16.5], [-31.1,21.1], [-13.9,26.5], [-17.8,30]]
rpos = [[coord[1],coord[0]] for coord in latlon]
nx.draw(rg, rpos, width = rwidths, node_color = cm(2/3))
plt.title('International Collaborations\nAcross Southern Africa', loc = 'center', fontsize = 20, color = 'black')
plt.savefig('C:/Users/User/Documents/Data/cpip_tests/collaborations_example.eps')
plt.show()


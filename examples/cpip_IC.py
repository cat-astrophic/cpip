# This script performs the core-periphery analysis of international collaboration
# using the MDPI data from Cary & Rockwell, 2020 for paper:
# Core-Periphery Models via Integer Programming: Maximizing the Influence of the Core

# Importing required modules

import pandas as pd
import numpy as np
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

# Plotting this network on a world political map

data = pd.read_csv(data_path + files[-1]) # reading in the data



# plot the core on a world political map??



# This script compares cpip to existing methods (max clique)

# Importing required modules

from cpip import *
import numpy as np
import networkx as nx
from matplotlib import pyplot as plt

# Importing benchmark networks from networkx

KCG = nx.karate_club_graph()
KKG = nx.krackhardt_kite_graph()

# Fixing vertex positions for consistent plotting

def drawer(cent):
    
    if cent < 5.68152066e-03:
        
        cent = 5.68152066e-03
        
    theta = 2 * np.pi * np.random.rand(1)[0]
    r = np.sqrt(1 / cent)
    a = r * np.cos(theta)
    b = r * np.sin(theta)
    
    return [a,b]

ec_kc = nx.eigenvector_centrality_numpy(KCG)
nx.set_node_attributes(KCG, ec_kc, 'ec')
node_ec_kc = [float(KCG.nodes[v]['ec']) for v in KCG]
locs_kc = [drawer(node_ec_kc[x]) for x in range(len(KCG.nodes()))]
pos_kc = dict(zip(KCG.nodes(), locs_kc))

pos_kk = nx.spring_layout(KKG, seed = 41)
pos_kk2 = nx.spring_layout(KKG, seed = 85)

# Maximum clique method :: C_KC = {0,1,2,3,13}; C_KK = {0,2,3,5}

plt.rcParams.update(plt.rcParamsDefault)
plt.rcParams.update({'figure.figsize': (15, 10)})
kcg_cliques = list(nx.find_cliques(KCG))
kcg_max_clique = max(kcg_cliques, key = len)
kcg_node_color = [(0.5, 0.5, 0.5) for v in KCG.nodes()]

for i, v in enumerate(KCG.nodes()):
    
    if v in kcg_max_clique:
        
        kcg_node_color[i] = (1, 0.1, 0.1)

nx.draw_networkx(KCG, node_color = kcg_node_color, pos = pos_kc)

plt.rcParams.update(plt.rcParamsDefault)
plt.rcParams.update({'figure.figsize': (15, 10)})
kkg_cliques = list(nx.find_cliques(KKG))
kkg_max_clique = max(kkg_cliques, key = len)
kkg_node_color = [(0.5, 0.5, 0.5) for v in KKG.nodes()]

for i, v in enumerate(KKG.nodes()):
    
    if v in kkg_max_clique:
        
        kkg_node_color[i] = (1, 0.1, 0.1)

# Using cpip :: C_KC = {{},{0,2},{0,2,8,13,19,31,32,33}}; C_KK = {{5,6},{3,5,6,7},{0,1,2,3,4,5,6,7}}

kc_1630 = cpip(nx.to_numpy_array(KCG),1.6,3)
kc_1830 = cpip(nx.to_numpy_array(KCG),1.8,3)
kc_2030 = cpip(nx.to_numpy_array(KCG),2,3)

kk_1630 = cpip(nx.to_numpy_array(KKG),1.6,3)
kk_1830 = cpip(nx.to_numpy_array(KKG),1.8,3)
kk_2030 = cpip(nx.to_numpy_array(KKG),2,3)

kc_1630 = [int(v[1:]) for v in kc_1630]
kc_1830 = [int(v[1:]) for v in kc_1830]
kc_2030 = [int(v[1:]) for v in kc_2030]

kk_1630 = [int(v[1:]) for v in kk_1630]
kk_1830 = [int(v[1:]) for v in kk_1830]
kk_2030 = [int(v[1:]) for v in kk_2030]

plt.rcParams.update(plt.rcParamsDefault)
plt.rcParams.update({'figure.figsize': (15, 10)})
kc1630_node_color = [(0.5, 0.5, 0.5) for v in KCG.nodes()]

for i, v in enumerate(KCG.nodes()):
    
    if v in kc_1630:
        
        kc1630_node_color[i] = (1, 0.1, 0.1)

plt.rcParams.update(plt.rcParamsDefault)
plt.rcParams.update({'figure.figsize': (15, 10)})
kc1830_node_color = [(0.5, 0.5, 0.5) for v in KCG.nodes()]

for i, v in enumerate(KCG.nodes()):
    
    if v in kc_1830:
        
        kc1830_node_color[i] = (1, 0.1, 0.1)

plt.rcParams.update(plt.rcParamsDefault)
plt.rcParams.update({'figure.figsize': (15, 10)})
kc2030_node_color = [(0.5, 0.5, 0.5) for v in KCG.nodes()]

for i, v in enumerate(KCG.nodes()):
    
    if v in kc_2030:
        
        kc2030_node_color[i] = (1, 0.1, 0.1)

plt.rcParams.update(plt.rcParamsDefault)
plt.rcParams.update({'figure.figsize': (15, 10)})
kk1630_node_color = [(0.5, 0.5, 0.5) for v in KKG.nodes()]

for i, v in enumerate(KKG.nodes()):
    
    if v in kk_1630:
        
        kk1630_node_color[i] = (1, 0.1, 0.1)

plt.rcParams.update(plt.rcParamsDefault)
plt.rcParams.update({'figure.figsize': (15, 10)})
kk1830_node_color = [(0.5, 0.5, 0.5) for v in KKG.nodes()]

for i, v in enumerate(KKG.nodes()):
    
    if v in kk_1830:
        
        kk1830_node_color[i] = (1, 0.1, 0.1)

plt.rcParams.update(plt.rcParamsDefault)
plt.rcParams.update({'figure.figsize': (15, 10)})
kk2030_node_color = [(0.5, 0.5, 0.5) for v in KKG.nodes()]

for i, v in enumerate(KKG.nodes()):
    
    if v in kk_2030:
        
        kk2030_node_color[i] = (1, 0.1, 0.1)

# Creating figures

nx.draw_networkx(KCG, node_color = kcg_node_color, pos = pos_kc)
nx.draw_networkx(KKG, node_color = kkg_node_color, pos = pos_kk)

nx.draw_networkx(KCG, node_color = kc1630_node_color, pos = pos_kc)
nx.draw_networkx(KCG, node_color = kc1830_node_color, pos = pos_kc)
nx.draw_networkx(KCG, node_color = kc2030_node_color, pos = pos_kc)

nx.draw_networkx(KKG, node_color = kk1630_node_color, pos = pos_kk2)
nx.draw_networkx(KKG, node_color = kk1830_node_color, pos = pos_kk2)
nx.draw_networkx(KKG, node_color = kk2030_node_color, pos = pos_kk2)

nx.draw_networkx(KKG, node_color = kk1630_node_color, pos = pos_kk)
nx.draw_networkx(KKG, node_color = kk1830_node_color, pos = pos_kk)
nx.draw_networkx(KKG, node_color = kk2030_node_color, pos = pos_kk)

#####################################################################

# Communities stuff

#from networkx.algorithms.community.centrality import girvan_newman
#communities = girvan_newman(G)
#node_groups = []
#for com in next(communities):
#    node_groups.append(list(com))
#print(node_groups)
#color_map = []
#for node in G:
#    if node in node_groups[0]:
#        color_map.append("red")
#    else:
#        color_map.append("orange")
#nx.draw(G, node_color=color_map, with_labels=True)


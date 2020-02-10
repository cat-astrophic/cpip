# This function identifies the core of a network in a core-periphery analysis with integer programming via pulp

# The input must be a weighted adjacency matrix for a network, hence the input need not be symmetric (e.g., directed networks)

# Import required modules
    
import pulp
import numpy as np
import pandas as pd
import networkx as nx
import itertools as it
from matplotlib import pyplot as plt

####################################################################################################

# Defining the main function

def cpip(filepath, theta, psi):
    
    # Safety feature for user error
    
    theta = max(theta,1)
    psi = max(psi,1)
    
    # Read in the data set
    
    W = pd.read_csv(filepath)
    
    # Remove all isolated vertices from the data set to be safe
    
    isolates = [idx for idx in range(len(W)) if sum(W[W.columns[idx]]) == 0]
    
    for iso in [isolates[len(isolates)-1-i] for i in range(len(isolates))]:
        
        W = W.drop(W.columns[iso], axis = 1).drop(iso, axis = 0)
    
    W2 = W.set_index(pd.Index([i for i in range(len(W))]))
    cols = W.columns
    W = W.values
    
    # Create the vectors c and b
    
    c = np.diag(np.matmul(W,np.ones((len(W),len(W)))))
    b = (len(W)-1)*np.ones(len(W))
    
    # Create the matrix A
    
    # Create a binary adjacency matrix
    
    M = np.zeros((len(W),len(W)))
    
    for row in range(len(W)):
        
        for col in range(len(W)):
            
            if W[row][col] > 0:
                
                M[row][col] = 1
   
    # Feed this matrix into networkx and create the distance matrix D
    
    G = nx.Graph(M)
    D = np.zeros((len(W),len(W)))
    
    for row in range(len(D)):
        
        for col in range(len(D)):
            
            if (col > row) and D[row][col] == 0:
                
                D[row][col] = nx.shortest_path_length(G, row, col)
    
    D = D + np.transpose(D)    
    A = np.diag(np.matmul(D,np.ones(len(W))))
    
    # Solve the program
    
    problem = pulp.LpProblem('Core-Periphery Network Model', pulp.LpMaximize)
        
    # Initialize a list of choice variables
    
    x = [pulp.LpVariable(col, lowBound = 0, upBound = 1, cat = 'Integer') for col in cols]
    
    # Define the objective function
    
    problem += pulp.lpSum([c[i]*x[i] for i in range(len(c))])
    
    # Constraints
    
    for row in range(len(A)):
        
        problem += pulp.lpSum([A[row][i]*x[i] for i in range(len(A))]) <= theta*b[row]
        
    # Solve this problem
    
    problem.solve()
    
    # Create the truncated data
    
    # Create a reference list of nations which were in the solution to part 1
    
    subset = []
    
    for var in problem.variables():
        
        if var.varValue > 0:
            
            subset.append(str(var).replace('_', ' '))
        
    # Continue this process if the first stage optimization problem yields at least two potential members of the core

    if (len(subset) < 2) or (theta == 1):
        
        return subset
    
    else:
        
        # Iteratively create new networks and solve corresponding optimization problems
        
        idxset = [list(W2.columns).index(sub) for sub in subset] # indices of core candidate vertices
        dropset = [b for b in range(len(W2.columns)) if b not in idxset] # remaining indices
        Dx = pd.DataFrame(D, columns = W2.columns)
        
        for idx in [dropset[len(dropset)-1-i] for i in range(len(dropset))]:

            Dx = Dx.drop(Dx.columns[idx], axis = 1).drop(idx, axis = 0)

        # Running the loop to find the optimal core
        
        val = 0
        core = []
        
        for a in range(1,len(subset)+1):
            
            # Generate all a-tuples
            
            tuples = list(it.combinations(idxset, a))
            
            # Check all combinations
            
            for t in tuples:
                
                tuple_ids = [t_id for t_id in t]
                drops = [k for k in list(Dx.index.values) if k not in tuple_ids]
                Dxa = Dx
                cxa = list(c)
                
                # Create the new matrices and vectors
                
                for ids in [drops[len(drops)-1-i] for i in range(len(drops))]:
                    
                    Dxa = Dxa.drop(W2.columns[ids], axis = 1).drop(ids, axis = 0)
                    cxa.pop(ids)
                
                Axa = Dxa.values                
                bxa = np.ones(len(Axa))
                
                # Define the problem
                
                prob = pulp.LpProblem('Core-Periphery Network Model', pulp.LpMaximize)
                
                # Initialize a list of choice variables
                
                y = [pulp.LpVariable(col, lowBound = 0, upBound = 1, cat = 'Integer') for col in Dxa.columns]
                
                # Define the objective function
                
                prob += pulp.lpSum([cxa[i]*y[i] for i in range(len(Dxa))])
                
                # Constraints
                
                for row in range(len(Axa)):
                
                    prob += pulp.lpSum([Axa[row][i]*y[i] for i in range(len(Axa))]) <= a*psi*bxa[row]
                
                # Solve the problem
                
                prob.solve()
                
                # Check to see if this is the new optimum
                
                if str(type(pulp.value(prob.objective))) != "<class 'NoneType'>":
                    
                    if pulp.value(prob.objective) > val:
                        
                        val, core = pulp.value(prob.objective), [str(v) for v in prob.variables() if v.varValue > 0]
                        
        # Update names elements in core to match original format and return core
        
        core = [c.replace('_', ' ') for c in core]
        
        return core

####################################################################################################

# Defining the exploratory function

def cpip_exploratory(filepath, theta):
    
    # Safety feature for user error
    
    theta = max(theta,1)
    
    # Read in the data set
    
    W = pd.read_csv(filepath)
    
    # Remove all isolated vertices from the data set to be safe
    
    isolates = [idx for idx in range(len(W)) if sum(W[W.columns[idx]]) == 0]
    
    for iso in [isolates[len(isolates)-1-i] for i in range(len(isolates))]:
        
        W = W.drop(W.columns[iso], axis = 1).drop(iso, axis = 0)
    
    cols = W.columns
    W = W.values
    
    # Create the vectors c and b
    
    c = np.diag(np.matmul(W,np.ones((len(W),len(W)))))
    b = (len(W)-1)*np.ones(len(W))
    
    # Create the matrix A
    
    # Create a binary adjacency matrix
    
    M = np.zeros((len(W),len(W)))
    
    for row in range(len(W)):
        
        for col in range(len(W)):
            
            if W[row][col] > 0:
                
                M[row][col] = 1
   
    # Feed this matrix into networkx and create the distance matrix D
    
    G = nx.Graph(M)
    D = np.zeros((len(W),len(W)))
    
    for row in range(len(D)):
        
        for col in range(len(D)):
            
            if (col > row) and D[row][col] == 0:
                
                D[row][col] = nx.shortest_path_length(G, row, col)
    
    D = D + np.transpose(D)    
    A = np.diag(np.matmul(D,np.ones(len(W))))
    
    # Solve the program
    
    problem = pulp.LpProblem('Core-Periphery Network Model', pulp.LpMaximize)
        
    # Initialize a list of choice variables
    
    x = [pulp.LpVariable(col, lowBound = 0, upBound = 1, cat = 'Integer') for col in cols]
    
    # Define the objective function
    
    problem += pulp.lpSum([c[i]*x[i] for i in range(len(c))])
    
    # Constraints
    
    for row in range(len(A)):
        
        problem += pulp.lpSum([A[row][i]*x[i] for i in range(len(A))]) <= theta*b[row]
        
    # Solve this problem
    
    problem.solve()
    
    # Create the truncated data
    
    # Create a reference list of nations which were in the solution to part 1
    
    subset = []
    
    for var in problem.variables():
        
        if var.varValue > 0:
            
            subset.append(str(var).replace('_', ' '))
        
    return subset

####################################################################################################

# Creating a class for the cpip_viz and cpip_stats functions

class network_object():
    
    pass

####################################################################################################

# Visualization function

def cpip_viz(filepath, core, *core_labels): # Note that core must be a list even if the core is empty or contains only one member
    
    # Read in the network data for the full network
    
    W = pd.read_csv(filepath)
    
    # Handling core_labels
    
    if len(core_labels) < 1:
        
        core_labels = True
    
    # Create the core and periphery of the network
    
    c_ids = [list(W.columns).index(c) for c in core]
    p_ids = [i for i in range(len(W.columns)) if W.columns[i] not in core]
    C = W
    P = W
    
    for c in [c_ids[len(c_ids)-1-i] for i in range(len(c_ids))]:
        
        P = P.drop(P.columns[c], axis = 1).drop(c, axis = 0)
    
    for p in [p_ids[len(p_ids)-1-i] for i in range(len(p_ids))]:
        
        C = C.drop(C.columns[p], axis = 1).drop(p, axis = 0)
    
    # Create and display the network, its core, and the periphery
    
    net_graph = nx.Graph(W.values)
    core_graph = nx.Graph(C.values)
    peri_graph = nx.Graph(P.values)
    
    plt.figure()
    nx.draw_circular(net_graph)

    if core_labels == True:
        
        core_pos = nx.circular_layout(core_graph)

        for c, p in core_pos.items(): # Labels for the core
            
            core_pos[c] = (p[0], p[1]+.25*((-1)**(round(c / len(core)))))
        
        if len(core) > 2:
        
            plt.figure()
            nx.draw_circular(core_graph)
            nx.draw_networkx_labels(core_graph, core_pos, dict(zip([i for i in range(len(core))],core)))
            plt.margins(.25)
            
        else: # Handling a weird networkx bug where it won't dispaly labels for graphs with a core containing less than 3 members
            
            plt.figure()
            nx.draw_circular(core_graph)
            
            if len(core) < 2:
                
                plt.text(0, .01, core[0], horizontalalignment = 'center')
            
            else:
                
                plt.text(1, .01, core[0], horizontalalignment = 'center')
                plt.text(-1, .01, core[1], horizontalalignment = 'center')

            plt.margins(.25)
        
    else:
        
        plt.figure()
        nx.draw_circular(core_graph)
    
    plt.figure()
    nx.draw_circular(peri_graph)
                            
    # Creating the output object
    
    output = network_object()
    output.network = network_object()
    output.network.data = W
    output.network.viz = net_graph
    output.core = network_object()
    output.core.data = C
    output.core.viz = core_graph
    output.periphery = network_object()
    output.periphery.data = P
    output.periphery.viz = peri_graph
    
    return output

####################################################################################################


# Function to generate some relevant statistics on the network and its core and periphery

def cpip_stats(filepath, core):
    
    # Read in the network data for the full network
    
    W = pd.read_csv(filepath)
    
    # Create the core and periphery of the network
    
    c_ids = [list(W.columns).index(c) for c in core]
    p_ids = [i for i in range(len(W.columns)) if W.columns[i] not in core]
    C = W
    P = W
    
    for c in [c_ids[len(c_ids)-1-i] for i in range(len(c_ids))]:
        
        P = P.drop(P.columns[c], axis = 1).drop(c, axis = 0)
    
    for p in [p_ids[len(p_ids)-1-i] for i in range(len(p_ids))]:
        
        C = C.drop(C.columns[p], axis = 1).drop(p, axis = 0)
    
    # Preparing some arrays of binary adjacency matrices for generating function outputs
    
    M = W.values
    MC = C.values
    MP = P.values
    
    for row in range(len(M)):
        
        for col in range(len(M)):
            
            if W.values[row][col] > 0:
                
                M[row][col] = 1
    
    for row in range(len(C)):
        
        for col in range(len(C)):
            
            if C.values[row][col] > 0:
                
                MC[row][col] = 1
    
    for row in range(len(P)):
        
        for col in range(len(P)):
            
            if P.values[row][col] > 0:
                
                MP[row][col] = 1
    
    output = network_object()
    output.core_size = len(c_ids)
    output.core_ratio = len(c_ids) / (len(c_ids) + len(p_ids))
    output.edges_network = sum(sum(M)) / 2
    output.edges_core = sum(sum(MC)) / 2
    output.edges_periphery = sum(sum(MP)) / 2
    output.edges_core_periphery = output.edges_network - output.edges_core - output.edges_periphery
    output.density_core = output.edges_core / ( ( len(MC) * (len(MC)-1) ) / 2 )
    output.density_periphery = output.edges_periphery / ( ( len(MP) * (len(MP)-1) ) / 2 )
    output.density_network = output.edges_network / ( ( len(M) * (len(M)-1) ) / 2 )
    output.average_degree_network = sum(sum(M)) / len(M)
    output.average_degree_core = (output.edges_core_periphery + output.edges_core) / len(C)
    output.average_degree_periphery = (output.edges_core_periphery + output.edges_periphery) / len(P)
        
    return output


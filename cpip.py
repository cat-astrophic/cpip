# This function identifies the core of a network in a core-periphery analysis with integer programming via pulp

# The input must be a (weighted) adjacency matrix for a network

# Import required modules

import pulp as __pulp__
import numpy as __np__
import pandas as __pd__
import networkx as __nx__
import itertools as __it__
from matplotlib import pyplot as __plt__

####################################################################################################

# Creating a helper function for reformatting column names to meet pulp specifications

def __pulp_names__(string):
    
    try:
        
        string_list = list(string)
        
    except:
        
        string_list = 'x' + str(string)
        
    for s in range(len(string_list)):
        
        if string_list[s].isalnum() == False:
            
            string_list[s] = '_'
    
    new_string = ''.join(string_list)
    
    return new_string

####################################################################################################

# Defining the main function

def cpip(A, theta, psi, loops = None):
    
    # Safety feature for user error
    
    theta = max(theta,1)
    psi = max(psi,1)
    
    # Convert numpy matrix to pandas dataframe
    
    W = __pd__.DataFrame(A)
    
    # Rename columns to match pulp formatting
    
    cc = list(W.columns)
    
    for c in range(len(cc)):
        
        cc[c] = __pulp_names__(cc[c])
        
    W.columns = cc
    
    # Remove loops (self interactions) if loops != True
    
    if loops != True:
        
        for i in range(len(W)):
            
            W[W.columns[i]][i] = 0
    
    # Ensure that the network is connected - find largest connected component
    
    G = __nx__.Graph(W.values)
    keep = list(max(__nx__.connected_components(G), key = len))
    remove = [i for i in range(len(G)) if i not in keep]
    
    for r in [remove[len(remove)-1-i] for i in range(len(remove))]:
        
        W = W.drop(W.columns[r], axis = 1).drop(r, axis = 0)
    
    W2 = W.set_index(__pd__.Index([i for i in range(len(W))]))
    cols = W.columns
    W = W.values
    
    # Create the vectors c and b
    
    c = __np__.diag(__np__.matmul(W,__np__.ones((len(W),len(W)))))
    b = (len(W)-1)*__np__.ones(len(W))
    
    # Create the matrix A
    
    # Create a binary adjacency matrix
    
    M = __np__.zeros((len(W),len(W)))
    
    for row in range(len(W)):
        
        for col in range(len(W)):
            
            if W[row][col] > 0:
                
                M[row][col] = 1
   
    # Feed this matrix into networkx and create the distance matrix D
    
    G = __nx__.Graph(M)
    D = __np__.zeros((len(W),len(W)))
    
    for row in range(len(D)):
        
        for col in range(len(D)):
            
            if (col > row) and D[row][col] == 0:
                
                D[row][col] = __nx__.shortest_path_length(G, row, col)
    
    D = D + __np__.transpose(D)
    A = __np__.diag(__np__.matmul(D,__np__.ones(len(W))))
    
    # Solve the program
    
    problem = __pulp__.LpProblem('Core-Periphery Network Model', __pulp__.LpMaximize)
        
    # Initialize a list of choice variables
    
    x = [__pulp__.LpVariable(col, lowBound = 0, upBound = 1, cat = 'Integer') for col in cols]
    
    # Define the objective function
    
    problem += __pulp__.lpSum([c[i]*x[i] for i in range(len(c))])
    
    # Constraints
    
    for row in range(len(A)):
        
        problem += __pulp__.lpSum([A[row][i]*x[i] for i in range(len(A))]) <= theta*b[row]
        
    # Solve this problem
    
    problem.solve()
    
    # Create the truncated data
    
    # Create a reference list of nations which were in the solution to part 1
    
    subset = []
    
    for var in problem.variables():
        
        if var.varValue > 0:
            
            subset.append(str(var))
        
    # Continue this process if the first stage optimization problem yields at least two potential members of the core

    if (len(subset) < 2) or (theta == 1):
        
        return subset
    
    else:
        
        # Iteratively create new networks and solve corresponding optimization problems
        
        idxset = [list(W2.columns).index(sub) for sub in subset] # indices of core candidate vertices
        dropset = [b for b in range(len(W2.columns)) if b not in idxset] # remaining indices
        Dx = __pd__.DataFrame(D, columns = W2.columns)
        
        for idx in [dropset[len(dropset)-1-i] for i in range(len(dropset))]:

            Dx = Dx.drop(Dx.columns[idx], axis = 1).drop(idx, axis = 0)

        # Running the loop to find the optimal core
        
        val = 0
        core = []
        
        for a in range(1,len(subset)+1):
            
            # Generate all a-tuples
            
            tuples = list(__it__.combinations(idxset, a))
            
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
                bxa = __np__.ones(len(Axa))
                
                # Define the problem
                
                prob = __pulp__.LpProblem('Core-Periphery Network Model', __pulp__.LpMaximize)
                
                # Initialize a list of choice variables
                
                y = [__pulp__.LpVariable(col, lowBound = 0, upBound = 1, cat = 'Integer') for col in Dxa.columns]
                
                # Define the objective function
                
                prob += __pulp__.lpSum([cxa[i]*y[i] for i in range(len(Dxa))])
                
                # Constraints
                
                for row in range(len(Axa)):
                
                    prob += __pulp__.lpSum([Axa[row][i]*y[i] for i in range(len(Axa))]) <= (a-1)*psi*bxa[row]
                
                # Solve the problem
                
                prob.solve()
                
                # Check to see if this is the new optimum
                
                if str(type(__pulp__.value(prob.objective))) != "<class 'NoneType'>":

                    if __pulp__.value(prob.objective) > val and len([str(v) for v in prob.variables() if v.varValue > 0]) == a:

                        val, core = __pulp__.value(prob.objective), [str(v) for v in prob.variables() if v.varValue > 0]
        
        return core

# Defining a version that reads in a csv from a given filepath

def cpip_fp(filepath, theta, psi, loops = None):
    
    # Safety feature for user error
    
    theta = max(theta,1)
    psi = max(psi,1)
    
    # Read in the data set
    
    W = __pd__.read_csv(filepath)
    
    # Rename columns to match pulp formatting
    
    cc = list(W.columns)
    
    for c in range(len(cc)):
        
        cc[c] = __pulp_names__(cc[c])
        
    W.columns = cc
    
    # Remove loops (self interactions) if loops != True
    
    if loops != True:
        
        for i in range(len(W)):
            
            W[W.columns[i]][i] = 0
    
    # Ensure that the network is connected - find largest connected component
    
    G = __nx__.Graph(W.values)
    keep = list(max(__nx__.connected_components(G), key = len))
    remove = [i for i in range(len(G)) if i not in keep]
    
    for r in [remove[len(remove)-1-i] for i in range(len(remove))]:
        
        W = W.drop(W.columns[r], axis = 1).drop(r, axis = 0)
    
    W2 = W.set_index(__pd__.Index([i for i in range(len(W))]))
    cols = W.columns
    W = W.values
    
    # Create the vectors c and b
    
    c = __np__.diag(__np__.matmul(W,__np__.ones((len(W),len(W)))))
    b = (len(W)-1)*__np__.ones(len(W))
    
    # Create the matrix A
    
    # Create a binary adjacency matrix
    
    M = __np__.zeros((len(W),len(W)))
    
    for row in range(len(W)):
        
        for col in range(len(W)):
            
            if W[row][col] > 0:
                
                M[row][col] = 1
   
    # Feed this matrix into networkx and create the distance matrix D
    
    G = __nx__.Graph(M)
    D = __np__.zeros((len(W),len(W)))
    
    for row in range(len(D)):
        
        for col in range(len(D)):
            
            if (col > row) and D[row][col] == 0:
                
                D[row][col] = __nx__.shortest_path_length(G, row, col)
    
    D = D + __np__.transpose(D)
    A = __np__.diag(__np__.matmul(D,__np__.ones(len(W))))
    
    # Solve the program
    
    problem = __pulp__.LpProblem('Core-Periphery Network Model', __pulp__.LpMaximize)
        
    # Initialize a list of choice variables
    
    x = [__pulp__.LpVariable(col, lowBound = 0, upBound = 1, cat = 'Integer') for col in cols]
    
    # Define the objective function
    
    problem += __pulp__.lpSum([c[i]*x[i] for i in range(len(c))])
    
    # Constraints
    
    for row in range(len(A)):
        
        problem += __pulp__.lpSum([A[row][i]*x[i] for i in range(len(A))]) <= theta*b[row]
        
    # Solve this problem
    
    problem.solve()
    
    # Create the truncated data
    
    # Create a reference list of nations which were in the solution to part 1
    
    subset = []
    
    for var in problem.variables():
        
        if var.varValue > 0:
            
            subset.append(str(var))
        
    # Continue this process if the first stage optimization problem yields at least two potential members of the core

    if (len(subset) < 2) or (theta == 1):
        
        return subset
    
    else:
        
        # Iteratively create new networks and solve corresponding optimization problems
        
        idxset = [list(W2.columns).index(sub) for sub in subset] # indices of core candidate vertices
        dropset = [b for b in range(len(W2.columns)) if b not in idxset] # remaining indices
        Dx = __pd__.DataFrame(D, columns = W2.columns)
        
        for idx in [dropset[len(dropset)-1-i] for i in range(len(dropset))]:

            Dx = Dx.drop(Dx.columns[idx], axis = 1).drop(idx, axis = 0)

        # Running the loop to find the optimal core
        
        val = 0
        core = []
        
        for a in range(1,len(subset)+1):
            
            # Generate all a-tuples
            
            tuples = list(__it__.combinations(idxset, a))
            
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
                bxa = __np__.ones(len(Axa))
                
                # Define the problem
                
                prob = __pulp__.LpProblem('Core-Periphery Network Model', __pulp__.LpMaximize)
                
                # Initialize a list of choice variables
                
                y = [__pulp__.LpVariable(col, lowBound = 0, upBound = 1, cat = 'Integer') for col in Dxa.columns]
                
                # Define the objective function
                
                prob += __pulp__.lpSum([cxa[i]*y[i] for i in range(len(Dxa))])
                
                # Constraints
                
                for row in range(len(Axa)):
                
                    prob += __pulp__.lpSum([Axa[row][i]*y[i] for i in range(len(Axa))]) <= (a-1)*psi*bxa[row]
                
                # Solve the problem
                
                prob.solve()
                
                # Check to see if this is the new optimum
                
                if str(type(__pulp__.value(prob.objective))) != "<class 'NoneType'>":

                    if __pulp__.value(prob.objective) > val and len([str(v) for v in prob.variables() if v.varValue > 0]) == a:

                        val, core = __pulp__.value(prob.objective), [str(v) for v in prob.variables() if v.varValue > 0]
        
        return core

####################################################################################################

# Defining the exploratory function

def cpip_exploratory(filepath, theta, loops = None):
    
    # Safety feature for user error
    
    theta = max(theta,1)
    
    # Read in the data set
    
    W = __pd__.read_csv(filepath)
    
    # Rename columns to match pulp formatting
    
    cc = list(W.columns)
    
    for c in range(len(cc)):
        
        cc[c] = __pulp_names__(cc[c])
        
    W.columns = cc    
    
    # Remove loops (self interactions) if loops != True
    
    if loops != True:
        
        for i in range(len(W)):
            
            W[W.columns[i]][i] = 0
    
    # Ensure that the network is connected - find largest connected component
    
    G = __nx__.Graph(W.values)
    keep = list(max(__nx__.connected_components(G), key = len))
    remove = [i for i in range(len(G)) if i not in keep]
    
    for r in [remove[len(remove)-1-i] for i in range(len(remove))]:
        
        W = W.drop(W.columns[r], axis = 1).drop(r, axis = 0)
    
    cols = W.columns
    W = W.values
    
    # Create the vectors c and b
    
    c = __np__.diag(__np__.matmul(W,__np__.ones((len(W),len(W)))))
    b = (len(W)-1)*__np__.ones(len(W))
    
    # Create the matrix A
    
    # Create a binary adjacency matrix
    
    M = __np__.zeros((len(W),len(W)))
    
    for row in range(len(W)):
        
        for col in range(len(W)):
            
            if W[row][col] > 0:
                
                M[row][col] = 1
   
    # Feed this matrix into networkx and create the distance matrix D
    
    G = __nx__.Graph(M)
    D = __np__.zeros((len(W),len(W)))
    
    for row in range(len(D)):
        
        for col in range(len(D)):
            
            if (col > row) and D[row][col] == 0:
                
                D[row][col] = __nx__.shortest_path_length(G, row, col)
    
    D = D + __np__.transpose(D)    
    A = __np__.diag(__np__.matmul(D,__np__.ones(len(W))))
    
    # Solve the program
    
    problem = __pulp__.LpProblem('Core-Periphery Network Model', __pulp__.LpMaximize)
    
    # Initialize a list of choice variables
    
    x = [__pulp__.LpVariable(col, lowBound = 0, upBound = 1, cat = 'Integer') for col in cols]
    
    # Define the objective function
    
    problem += __pulp__.lpSum([c[i]*x[i] for i in range(len(c))])
    
    # Constraints
    
    for row in range(len(A)):
        
        problem += __pulp__.lpSum([A[row][i]*x[i] for i in range(len(A))]) <= theta*b[row]
        
    # Solve this problem
    
    problem.solve()
    
    # Create the truncated data
    
    # Create a reference list of nations which were in the solution to part 1
    
    subset = []
    
    for var in problem.variables():
        
        if var.varValue > 0:
            
            subset.append(str(var))
        
    return subset

####################################################################################################

# Creating a class used in the cpip_viz function

class __network_object__():
    
    pass

####################################################################################################

# Visualization function

def cpip_viz(filepath, core, core_labels = None, savefigs = None, newpath = None):
    
    # Ensure that core is a list and not a string
    
    if len(core) == 1:
        
        core = list(core)
    
    # Read in the network data for the full network
    
    W = __pd__.read_csv(filepath)
    
    # Rename columns to match pulp formatting
    
    cc = list(W.columns)
    
    for c in range(len(cc)):
        
        cc[c] = __pulp_names__(cc[c])
        
    W.columns = cc
    
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
    
    net_graph = __nx__.Graph(W.values)
    core_graph = __nx__.Graph(C.values)
    peri_graph = __nx__.Graph(P.values)
    
    __plt__.figure()
    __nx__.draw_circular(net_graph)
    
    if savefigs != None:
        
        if newpath != None:
            
            __plt__.savefig(newpath + '_network.' + savefigs)

        else:
            
            __plt__.savefig(filepath[0:len(filepath)-4] + '_network.' + savefigs)

    if core_labels == True:
        
        core_pos = __nx__.circular_layout(core_graph)

        for c, p in core_pos.items(): # Labels for the core
            
            core_pos[c] = (p[0], p[1]+.25*((-1)**(round(c / len(core)))))
        
        if len(core) > 2:
        
            __plt__.figure()
            __nx__.draw_circular(core_graph)
            __nx__.draw_networkx_labels(core_graph, core_pos, dict(zip([i for i in range(len(core))],core)))
            __plt__.margins(.25)
            
        else: # Handling a weird networkx bug where it won't dispaly labels for graphs with a core containing less than 3 members
            
            __plt__.figure()
            __nx__.draw_circular(core_graph)
            
            if len(core) < 2:
                
                __plt__.text(0, .01, core[0], horizontalalignment = 'center')
            
            else:
                
                __plt__.text(1, .01, core[0], horizontalalignment = 'center')
                __plt__.text(-1, .01, core[1], horizontalalignment = 'center')

            __plt__.margins(.25)
        
        if savefigs != None:
            
            if newpath != None:
                
                __plt__.savefig(newpath + '_core.' + savefigs)
                
            else:
                
                __plt__.savefig(filepath[0:len(filepath)-4] + '_core.' + savefigs)
        
    else:
        
        __plt__.figure()
        __nx__.draw_circular(core_graph)
        
        if savefigs != None:
            
            if newpath != None:
                
                __plt__.savefig(newpath + '_core.' + savefigs)
                
            else:
                
                __plt__.savefig(filepath[0:len(filepath)-4] + '_core.' + savefigs)
    
    __plt__.figure()
    __nx__.draw_circular(peri_graph)
    
    if savefigs != None:
        
        if newpath != None:
                
            __plt__.savefig(newpath + '_periphery.' + savefigs)
                
        else:
                
            __plt__.savefig(filepath[0:len(filepath)-4] + '_periphery.' + savefigs)

####################################################################################################

# Function to generate some relevant statistics on the network and its core and periphery

def cpip_stats(filepath, core):
    
    # Read in the network data for the full network
    
    W = __pd__.read_csv(filepath)
    
    # Rename columns to match pulp formatting
    
    cc = list(W.columns)
    
    for c in range(len(cc)):
        
        cc[c] = __pulp_names__(cc[c])
        
    W.columns = cc
    
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
    
    # Creating the output object
    
    output = __network_object__()
    
    # Number of vertices
    
    output.order = __network_object__()
    output.order.network = len(M)
    output.order.core = len(c_ids)
    output.order.periphery = len(p_ids)
    
    # Number of edges
    
    output.size = __network_object__()
    output.size.network = sum(sum(M)) / 2
    output.size.core = sum(sum(MC)) / 2
    output.size.periphery = sum(sum(MP)) / 2
    output.size.between = output.size.network - output.size.core - output.size.periphery
    
    # Ratios of order and size
    
    output.ratio_v = __network_object__()
    output.ratio_v.network = 1
    output.ratio_v.core = output.order.core / output.order.network
    output.ratio_v.periphery = 1 - output.ratio_v.core
    
    output.ratio_e = __network_object__()
    output.ratio_e.network = 1
    output.ratio_e.core = output.size.core / output.size.network
    output.ratio_e.periphery = output.size.periphery / output.size.network
    output.ratio_e.between = 1 - output.ratio_e.core - output.ratio_e.periphery
    
    # Densities
    
    output.density = __network_object__()
    output.density.network = output.size.network / ( ( len(M) * (len(M)-1) ) / 2 )
    output.density.core = output.size.core / ( ( len(MC) * (len(MC)-1) ) / 2 )
    output.density.periphery = output.size.periphery / ( ( len(MP) * (len(MP)-1) ) / 2 )
    
    # Degree statistics (mean, min, max)
    
    output.within_degrees = __network_object__()
    output.total_degrees = __network_object__()
    
    output.within_degrees.average = __network_object__()
    output.within_degrees.average.network = sum(sum(M)) / len(M)
    output.within_degrees.average.core = sum(sum(MC)) / len(MC)
    output.within_degrees.average.periphery = sum(sum(MP)) / len(MP)
    
    output.within_degrees.min = __network_object__()
    output.within_degrees.min.network = min(sum(M))
    output.within_degrees.min.core = min(sum(MC))
    output.within_degrees.min.periphery = min(sum(MP))
    
    output.within_degrees.max = __network_object__()
    output.within_degrees.max.network = max(sum(M))
    output.within_degrees.max.core = max(sum(MC))
    output.within_degrees.max.periphery = max(sum(MP))
    
    output.total_degrees.average = __network_object__()
    output.total_degrees.average.network = output.within_degrees.average.network
    output.total_degrees.average.core = sum([sum(M)[c] for c in c_ids]) / len(c_ids)
    output.total_degrees.average.periphery = sum([sum(M)[p] for p in p_ids]) / len(p_ids)
    
    output.total_degrees.min = __network_object__()
    output.total_degrees.min.network = output.within_degrees.min.network
    output.total_degrees.min.core = min([sum(M)[c] for c in c_ids])
    output.total_degrees.min.periphery = min([sum(M)[p] for p in p_ids])

    output.total_degrees.max = __network_object__()
    output.total_degrees.max.network = output.within_degrees.max.network
    output.total_degrees.max.core = max([sum(M)[c] for c in c_ids])
    output.total_degrees.max.periphery = max([sum(M)[p] for p in p_ids])

    # Number of connected components
    
    output.components = __network_object__()
    output.components.network = __nx__.number_connected_components(__nx__.Graph(M))
    output.components.core = __nx__.number_connected_components(__nx__.Graph(MC))
    output.components.periphery = __nx__.number_connected_components(__nx__.Graph(MP))
    
    # Radius and diameter
    
    output.radius = __network_object__()
    output.diameter = __network_object__()
    
    if output.components.network != 1:
        
        output.radius.network = 'inf'
        output.diameter.network = 'inf'
        
    else:
        
        output.radius.network = __nx__.radius(__nx__.Graph(M))
        output.diameter.network = __nx__.diameter(__nx__.Graph(M))
    
    if output.components.core != 1:
        
        output.radius.core = 'inf'
        output.diameter.core = 'inf'
        
    else:
        
        output.radius.core = __nx__.radius(__nx__.Graph(MC))
        output.diameter.core = __nx__.diameter(__nx__.Graph(MC))
        
    if output.components.periphery != 1:
        
        output.radius.periphery = 'inf'
        output.diameter.periphery = 'inf'
        
    else:
        
        output.radius.periphery = __nx__.radius(__nx__.Graph(MP))
        output.diameter.periphery = __nx__.diameter(__nx__.Graph(MP))
    
    # Maximium clique size
    
    output.clique = __network_object__()
    output.clique.network = len(max(__nx__.find_cliques(__nx__.Graph(M))))
    output.clique.core = len(max(__nx__.find_cliques(__nx__.Graph(MC))))
    output.clique.periphery = len(max(__nx__.find_cliques(__nx__.Graph(MP))))
    
    # Global clustering
    
    output.clustering = __network_object__()
    output.clustering.network = __nx__.average_clustering(__nx__.Graph(M))
    output.clustering.core = __nx__.average_clustering(__nx__.Graph(MC))
    output.clustering.periphery = __nx__.average_clustering(__nx__.Graph(MP))
    
    # Connectivity statistics
    
    output.connectivity = __network_object__()
    output.edge_connectivity = __network_object__()
    output.algebraic_connectivity = __network_object__()
    
    if output.components.network > 1:
        
        output.connectivity.network = 0
        output.edge_connectivity.network = 0
        
    else:
        
        output.connectivity.network = __nx__.node_connectivity(__nx__.Graph(M))
        output.edge_connectivity.network = __nx__.edge_connectivity(__nx__.Graph(M))
    
    if output.components.core > 1:
        
        output.connectivity.core = 0
        output.edge_connectivity.core = 0
        
    else:
        
        output.connectivity.core = __nx__.node_connectivity(__nx__.Graph(MC))
        output.edge_connectivity.core = __nx__.edge_connectivity(__nx__.Graph(MC))
    
    if output.components.periphery > 1:
        
        output.connectivity.periphery = 0
        output.edge_connectivity.periphery = 0
        
    else:
        
        output.connectivity.periphery = __nx__.node_connectivity(__nx__.Graph(MP))
        output.edge_connectivity.periphery = __nx__.edge_connectivity(__nx__.Graph(MP))
    
    output.algebraic_connectivity.network = __nx__.algebraic_connectivity(__nx__.Graph(M))
    output.algebraic_connectivity.core = __nx__.algebraic_connectivity(__nx__.Graph(MC))
    output.algebraic_connectivity.periphery = __nx__.algebraic_connectivity(__nx__.Graph(MP))
    
    # Energies and Laplacian energies
    
    output.energy = __network_object__()
    output.laplacian_energy = __network_object__()
    
    output.energy.network = sum(abs(__nx__.adjacency_spectrum(__nx__.Graph(M))))
    output.energy.core = sum(abs(__nx__.adjacency_spectrum(__nx__.Graph(MC))))
    output.energy.periphery = sum(abs(__nx__.adjacency_spectrum(__nx__.Graph(MP))))
    
    output.laplacian_energy.network = sum(abs(__nx__.laplacian_spectrum(__nx__.Graph(M))))
    output.laplacian_energy.core = sum(abs(__nx__.laplacian_spectrum(__nx__.Graph(MC))))
    output.laplacian_energy.periphery = sum(abs(__nx__.laplacian_spectrum(__nx__.Graph(MP))))
    
    # Transitivity
    
    output.transitivity = __network_object__()
    output.transitivity.network = __nx__.transitivity(__nx__.Graph(M))
    output.transitivity.core = __nx__.transitivity(__nx__.Graph(MC))
    output.transitivity.periphery = __nx__.transitivity(__nx__.Graph(MP))
    
    # Wiener index
    
    output.wiener = __network_object__()
    output.wiener.network = __nx__.wiener_index(__nx__.Graph(M))
    output.wiener.core = __nx__.wiener_index(__nx__.Graph(MC))
    output.wiener.periphery = __nx__.wiener_index(__nx__.Graph(MP))
    
    # Check if the core is a dominating set
    
    output.dom_set = __network_object__()
    output.dom_set.core = __nx__.is_dominating_set(__nx__.Graph(M), c_ids)
    output.dom_set.periphery = __nx__.is_dominating_set(__nx__.Graph(M), p_ids)
    
    return output

####################################################################################################
    
# A helper function for printing the table in cpip_summary

def __table_helper__(ids):
    
    if ids == [0]:
        
        s = '{:<25s}{:^20.0f}{:^20.0f}{:^20s}'
        
    elif ids == [1]:
        
        s = '{:<25s}{:^20s}{:^20.0f}{:^20.0f}'
        
    elif ids == [2]:
        
        s = '{:<25s}{:^20.0f}{:^20s}{:^20.0f}'
        
    elif ids == [0,1]:
        
        s = '{:<25s}{:^20s}{:^20.0f}{:^20s}'
        
    elif ids == [0,2]:
        
        s = '{:<25s}{:^20.0f}{:^20s}{:^20s}'
        
    elif ids == [1,2]:
        
        s = '{:<25s}{:^20s}{:^20s}{:^20.0f}'
        
    elif ids == [0,1,2]:
        
        s = '{:<25s}{:^20s}{:^20s}{:^20s}'
    
    return s

####################################################################################################

# A function to print a table containing the data from cpip_stats

def cpip_summary(output):
    
    # Manipulate output into printable data
    
    results = [[output.order.network, output.order.core, output.order.periphery],
               [output.size.network, output.size.core, output.size.periphery],
               [output.size.between],
               [output.ratio_v.network, output.ratio_v.core, output.ratio_v.periphery],
               [output.ratio_e.network, output.ratio_e.core, output.ratio_e.periphery],
               [output.density.network, output.density.core, output.density.periphery],
               [output.total_degrees.average.network, output.total_degrees.average.core, output.total_degrees.average.periphery],
               [output.total_degrees.min.network, output.total_degrees.min.core, output.total_degrees.min.periphery],
               [output.total_degrees.max.network, output.total_degrees.max.core, output.total_degrees.max.periphery],
               [output.within_degrees.average.network, output.within_degrees.average.core, output.within_degrees.average.periphery],
               [output.within_degrees.min.network, output.within_degrees.min.core, output.within_degrees.min.periphery],
               [output.within_degrees.max.network, output.within_degrees.max.core, output.within_degrees.max.periphery],
               [output.components.network, output.components.core, output.components.periphery],
               [output.connectivity.network, output.connectivity.core, output.connectivity.periphery],
               [output.edge_connectivity.network, output.edge_connectivity.core, output.edge_connectivity.periphery],
               [output.algebraic_connectivity.network, output.algebraic_connectivity.core, output.algebraic_connectivity.periphery],
               [output.clustering.network, output.clustering.core, output.clustering.periphery],
               [output.clique.network, output.clique.core, output.clique.periphery],
               [output.dom_set.core, output.dom_set.periphery],
               [output.transitivity.network, output.transitivity.core, output.transitivity.periphery],
               [output.radius.network, output.radius.core, output.radius.periphery],
               [output.diameter.network, output.diameter.core, output.diameter.periphery],
               [output.energy.network, output.energy.core, output.energy.periphery],
               [output.laplacian_energy.network, output.laplacian_energy.core, output.laplacian_energy.periphery],
               [output.wiener.network, output.wiener.core, output.wiener.periphery]]
    
    # Create list of row names for table
    
    stat_names = ['Order', 'Size', 'Size (Core - Periphery)', 'Ratio - Order', 'Ratio - Size', 'Density',
                  'Average Degree - Total', 'Min Degree - Total', 'Max Degree - Total', 'Average Degree - Within', 
                  'Min Degree - Within', 'Max Degree - Within', '# Components', 'Connectivity', 'Edge Connectivity', 
                  'Algebraic Connectivity', 'Clustering Coefficient', 'Clique Number', 'Dominating Set', 'Transitivity', 
                  'Radius', 'Diameter', 'Energy', 'Laplacian Energy', 'Wiener Index']
    
    # Print the table
    
    line = '-' * 85
    header = ['Statistic', 'Core', 'Periphery', 'Network']
    
    print(line)
    print('{:^25s}{:^20s}{:^20s}{:^20s}'.format(header[0], header[1], header[2], header[3]))
    print(line)
    
    for i in range(len(results)):
        
        if stat_names[i] == 'Size (Core - Periphery)':
            
            print('{:<25s}{:^20.0f}{:^20.0f}{:^20s}'.format(stat_names[i], results[i][0], results[i][0], '---'))
            
        elif stat_names[i] == 'Dominating Set':
            
            print('{:<25s}{:^20s}{:^20s}{:^20s}'.format(stat_names[i], str(results[i][0]), str(results[i][1]), 'True'))
                       
        elif ('inf' in results[i]) and (stat_names[i] != 'Wiener Index'):
            
            ids = [idx for idx in range(3) if results[i][idx] == 'inf']
            print(__table_helper__(ids).format(stat_names[i], results[i][1], results[i][2], results[i][0]))
            
        elif stat_names[i] in ['Order', 'Size', 'Min Degree - Total', 'Max Degree - Total', 'Min Degree - Within',
                       'Max Degree - Within', '# Components', 'Clique Number', 'Connectivity', 'Edge Connectivity',
                       'Algebraic Connectivity']:
            
            print('{:<25}{:^20.0f}{:^20.0f}{:^20.0f}'.format(stat_names[i], results[i][1], results[i][2], results[i][0]))
            
        else:
            
            print('{:<25}{:^20.3f}{:^20.3f}{:^20.3f}'.format(stat_names[i], results[i][1], results[i][2], results[i][0]))
            
    print(line)

####################################################################################################
    
# A function to print a table with tex formatting containing the data from cpip_stats

def cpip_summary_tex(output):
    
    # Manipulate output into printable data
    
    results = [[output.order.network, output.order.core, output.order.periphery],
               [output.size.network, output.size.core, output.size.periphery],
               [output.size.between],
               [output.ratio_v.network, output.ratio_v.core, output.ratio_v.periphery],
               [output.ratio_e.network, output.ratio_e.core, output.ratio_e.periphery],
               [output.density.network, output.density.core, output.density.periphery],
               [output.total_degrees.average.network, output.total_degrees.average.core, output.total_degrees.average.periphery],
               [output.total_degrees.min.network, output.total_degrees.min.core, output.total_degrees.min.periphery],
               [output.total_degrees.max.network, output.total_degrees.max.core, output.total_degrees.max.periphery],
               [output.within_degrees.average.network, output.within_degrees.average.core, output.within_degrees.average.periphery],
               [output.within_degrees.min.network, output.within_degrees.min.core, output.within_degrees.min.periphery],
               [output.within_degrees.max.network, output.within_degrees.max.core, output.within_degrees.max.periphery],
               [output.components.network, output.components.core, output.components.periphery],
               [output.connectivity.network, output.connectivity.core, output.connectivity.periphery],
               [output.edge_connectivity.network, output.edge_connectivity.core, output.edge_connectivity.periphery],
               [output.algebraic_connectivity.network, output.algebraic_connectivity.core, output.algebraic_connectivity.periphery],
               [output.clustering.network, output.clustering.core, output.clustering.periphery],
               [output.clique.network, output.clique.core, output.clique.periphery],
               [output.dom_set.core, output.dom_set.periphery],
               [output.transitivity.network, output.transitivity.core, output.transitivity.periphery],
               [output.radius.network, output.radius.core, output.radius.periphery],
               [output.diameter.network, output.diameter.core, output.diameter.periphery],
               [output.energy.network, output.energy.core, output.energy.periphery],
               [output.laplacian_energy.network, output.laplacian_energy.core, output.laplacian_energy.periphery],
               [output.wiener.network, output.wiener.core, output.wiener.periphery]]
    
    # Create list of row names for table
    
    stat_names = ['Order', 'Size', 'Size (Core - Periphery)', 'Ratio - Order', 'Ratio - Size', 'Density',
                  'Average Degree - Total', 'Min Degree - Total', 'Max Degree - Total', 'Average Degree - Within', 
                  'Min Degree - Within', 'Max Degree - Within', '\# Components', 'Connectivity', 'Edge Connectivity', 
                  'Algebraic Connectivity', 'Clustering Coefficient', 'Clique Number', 'Dominating Set', 'Transitivity', 
                  'Radius', 'Diameter', 'Energy', 'Laplacian Energy', 'Wiener Index']
    
    # Print the tex formatted table
    
    header = ['Statistic', 'Core', 'Periphery', 'Network']
    
    print('\\begin{table}[h!]')
    print('\centering')
    print('\small')
    print('\caption{A table containing several key statistics for the network, the core, and the periphery.}')
    print('\\begin{tabular}{lccc}\hline\hline')
    print('\\rule{0pt}{3ex}')
    print(header[0] + ' & ' + header[1] + ' & ' + header[2] + ' & ' + header[3] + '\\\\\hline')
    print('\\rule{0pt}{3ex}')
    
    for i in range(len(results)):
        
        if stat_names[i] == 'Size (Core - Periphery)':
            
            print(stat_names[i] + ' & ' + str(results[i][0]) + ' & ' + str(results[i][0]) + ' & ---\\\\')
            print('\\rule{0pt}{3ex}')
            
        elif stat_names[i] == 'Dominating Set':
            
            print(stat_names[i] + ' & ' + str(results[i][0]) + ' & ' + str(results[i][1]) + ' & True\\\\')
            print('\\rule{0pt}{3ex}')
                       
        elif ('inf' in results[i]) and (stat_names[i] != 'Wiener Index'):
            
            print(stat_names[i] + ' & ' + str(results[i][1]) + ' & ' + str(results[i][2]) + ' & ' + str(results[i][0]) + '\\\\')
            print('\\rule{0pt}{3ex}')
            
        elif stat_names[i] in ['Order', 'Size', 'Min Degree - Total', 'Max Degree - Total', 'Min Degree - Within',
                       'Max Degree - Within', '# Components', 'Clique Number', 'Connectivity', 'Edge Connectivity',
                       'Algebraic Connectivity']:
            
            print(stat_names[i] + ' & ' + str(results[i][1]) + ' & ' + str(results[i][2]) + ' & ' + str(results[i][0]) + '\\\\')
            print('\\rule{0pt}{3ex}')
        
        elif stat_names[i] == 'Wiener Index':
            
            print(stat_names[i] + ' & ' + str(round(results[i][1],3)) + ' & ' + str(round(results[i][2],3)) + ' & ' + str(round(results[i][0],3)) + '\\\\\hline\hline')
        
        else:
            
            print(stat_names[i] + ' & ' + str(round(results[i][1],3)) + ' & ' + str(round(results[i][2],3)) + ' & ' + str(round(results[i][0],3)) + '\\\\')
            print('\\rule{0pt}{3ex}')
            
    print('\end{tabular}')
    print('\end{table}')


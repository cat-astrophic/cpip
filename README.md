## Core-Periphery Integer Program (cpip)

* cpip is an in-development project which identifies the core of a network by using a paramterized algorithmic approach which solves a series of integer programs

* cpip uses only two parameters: one which places a constraint on the centrality of the core within the network, and one which puts a constraint on the density of the core

* cpip requires the following modules
    * **pulp** for solving the integer programs
    * **networkx** for visualizing networks and computing network statistics
    * **matplotlib** for general plotting
    * **itertools** for data manipulation
    * **pandas** for data manipulation
    * **numpy** for data manipulation

* To use cpip within your own python code include the following line at the beginning of your code (note that for now you will need to save the file *cpip.py* to your working directory as several major add-ons are currently in the works):
   * from cpip import *

* Once that line of code is included you can find the core of a network by defining variable *filepath* which containts the location of a csv file containing the weighted adjacency matrix for your network, by specifying values for the parameters *theta* and *psi*, and then running the following line of code:
   * core = cpip(filepath, theta, psi)

* If your network has loops and you wish to consider this data, you can simply set *loops = True* as an argument in cpip:
   * core = cpip(filepath, theta, psi, loops = True)
 
* If you would like to visualize your network, its core, and its periphey, once you have run cpip you can use hte cpip_viz function:
   * cpip_viz(filename, core)

* If you want to add text labels to the visuzlization of the core, cpip_viz will let you use the column headings from your data file with the core_labels argument as follows:
   * cpip_viz(filepath, core, core_labels = True)

* If you are not sure about how you want to parameterize cpip, you can use the cpip_exploratory function to run only the first stage of the cpip algorithm and find the set of core candidates. This helps indicate how long cpip will run while also providing intuition about your network. Since this runs far more quickly than cpip itself, you can use this to help determine an appropriate value for theta.
   * core = cpip_exploratory(filepath, theta)

* If you would like to see a table of network statistics for the core, the periphery, and for the network as a whole:
   * output = cpip_stats(filepath, core)
   * cpip_summary(output)

* And if you want the table in tex formatting:
   * cpip_summary_tex(output)

<!-- * A paper introducing the method used in cpip along with the included examples is currently undergoing peer review at *Social Networks*. -->

<!-- solves an initial linear program (display below) -->

<!-- loops through (show loop) -->

<!-- reference to pulp? -->

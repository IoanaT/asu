from pgmpy.models import DynamicBayesianNetwork as DBN
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import DBNInference

# Construct a DBN object
dbn = DBN()

# Create the edges for this 2-TBN (two time slice bayesian network)
# For edges in the same time slice, you only need to provide their connections in the first slice
dbn.add_edges_from([(('D', 0),('G', 0)),(('I', 0),('G', 0)),(('D', 0),('D', 1)),(('I', 0),('I', 1))])

# Create the CPD (Conditional Probability Distribution) tables
# First, create the CPD tables for edges in the same time slice, i.e., the prior
d_cpd = TabularCPD(('D', 0), 2, [[0.6, 0.4]])
i_cpd = TabularCPD(('I', 0), 2, [[0.7, 0.3]])
g_cpd = TabularCPD(('G', 0), 3, [[0.3, 0.05, 0.9, 0.5],
                                      [0.4, 0.25, 0.08, 0.3],
                                      [0.3, 0.7, 0.02, 0.2]],
                        evidence=[('I', 0),('D', 0)],
                        evidence_card=[2, 2])

# Next, create the CPD tables for edges across time slices
d_i_cpd = TabularCPD(('D',1), 2, [[0.6, 0.3],
                                   [0.4, 0.7]],
                     evidence=[('D',0)],
                      evidence_card=[2])
i_i_cpd = TabularCPD(('I', 1), 2, [[0.5, 0.4],
                                   [0.5, 0.6]],
                      evidence=[('I', 0)],
                      evidence_card=[2])

# Add the CPD tables into the DBN
dbn.add_cpds(d_cpd, i_cpd, g_cpd, d_i_cpd, i_i_cpd)

# Print the nodes  in the DBN
print(dbn.nodes())

# Print the nodes at time slice 2
print(dbn.get_slice_nodes(2))

# Do NOT forget to initialize before doing any inference! Otherwise, errors will be introduced.
dbn.initialize_initial_state()

# Create an inference object
dbn_inf = DBNInference(dbn)

# Perform inference on the DBN.
print(dbn_inf.query(variables=[('G',3)], evidence={('G',0): 1, ('G',1): 2})[('G',3)])

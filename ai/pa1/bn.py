# Starting with defining the network structure
from pgmpy.models import BayesianModel
from pgmpy.factors.discrete import TabularCPD

# Define the model structure (also see the instructions)
cancer_model = BayesianModel([('Pollution', 'Cancer'), 
                              ('Smoker', 'Cancer'),
                              ('Cancer', 'Xray'),
                              ('Cancer', 'Dyspnoea')])

# now defining the parameters.
cpd_poll = TabularCPD(variable='Pollution', variable_card=2,
                      values=[[0.9], [0.1]])
cpd_smoke = TabularCPD(variable='Smoker', variable_card=2,
                       values=[[0.3], [0.7]])
cpd_cancer = TabularCPD(variable='Cancer', variable_card=2,
                        values=[[0.03, 0.05, 0.001, 0.02],
                                [0.97, 0.95, 0.999, 0.98]],
                        evidence=['Smoker', 'Pollution'],
                        evidence_card=[2, 2])
cpd_xray = TabularCPD(variable='Xray', variable_card=2,
                      values=[[0.9, 0.2], [0.1, 0.8]],
                      evidence=['Cancer'], evidence_card=[2])
cpd_dysp = TabularCPD(variable='Dyspnoea', variable_card=2,
                      values=[[0.65, 0.3], [0.35, 0.7]],
                      evidence=['Cancer'], evidence_card=[2])

# Associating the parameters with the model structure.
cancer_model.add_cpds(cpd_poll, cpd_smoke, cpd_cancer, cpd_xray, cpd_dysp)

# Checking if the cpds are valid for the model.
print(cancer_model.check_model())

# Check d-separations. This is only meant for those interested. You do not need to understand this to do the project.
print(cancer_model.is_active_trail('Pollution', 'Smoker'))
print(cancer_model.is_active_trail('Pollution', 'Smoker', observed=['Cancer']))
print(cancer_model.local_independencies('Xray'))
print(cancer_model.get_independencies())

# Print model information
print(cancer_model.edges())
print(cancer_model.nodes())
print(cancer_model.get_cpds())

# Doing exact inference using Variable Elimination
from pgmpy.inference import VariableElimination
cancer_infer = VariableElimination(cancer_model)

# Query
print(cancer_infer.query(variables=['Dyspnoea'], evidence={'Cancer': 0},joint=False)['Dyspnoea'])
print(cancer_infer.query(variables=['Cancer'], evidence={'Smoker': 0, 'Pollution': 0},joint=False)['Cancer'])


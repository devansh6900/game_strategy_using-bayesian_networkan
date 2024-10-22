# Import required libraries
from pgmpy.models import BayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination

# Step 1: Define the structure of the Bayesian Network
# We assume that the next move depends on the previous move and the opponent's health
model = BayesianNetwork([('Health', 'Next_Move'), 
                         ('Previous_Move', 'Next_Move')])

# Step 2: Define the CPDs (Conditional Probability Distributions)

# CPD for Health (H)
cpd_health = TabularCPD(variable='Health', variable_card=3, 
                        values=[[0.2],   # Low health
                                [0.5],   # Medium health
                                [0.3]])  # High health

# CPD for Previous Move (P)
cpd_previous_move = TabularCPD(variable='Previous_Move', variable_card=3, 
                               values=[[0.4],   # Attack
                                       [0.3],   # Defend
                                       [0.3]])  # Heal

# CPD for Next Move (N) given Health and Previous Move
# The table defines how the next move depends on health and the previous move
cpd_next_move = TabularCPD(variable='Next_Move', variable_card=3, 
                           values=[
                               [0.6, 0.7, 0.8, 0.3, 0.2, 0.4, 0.1, 0.1, 0.3],  # Attack
                               [0.3, 0.2, 0.1, 0.4, 0.5, 0.3, 0.4, 0.3, 0.5],  # Defend
                               [0.1, 0.1, 0.1, 0.3, 0.3, 0.3, 0.5, 0.6, 0.2]], # Heal
                           evidence=['Health', 'Previous_Move'],
                           evidence_card=[3, 3])

# Step 3: Add CPDs to the model
model.add_cpds(cpd_health, cpd_previous_move, cpd_next_move)

# Step 4: Check if the model is valid
assert model.check_model()

# Step 5: Perform inference using Variable Elimination
inference = VariableElimination(model)

# Step 6: Query the probability of the opponent's Next Move 
# given that the Health is Medium and the Previous Move was Attack
query_result = inference.query(variables=['Next_Move'], 
                               evidence={'Health': 1,   # Medium health
                                         'Previous_Move': 0})  # Previous move was Attack

# Display the result
print(query_result)

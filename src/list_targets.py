import pandas as pd
from src.chembl_api import find_targets_for_disease

# Fetch up to 10 targets related to 'malaria'
targets_df = find_targets_for_disease("malaria", max_results=10)

# Display the targets
print(targets_df)

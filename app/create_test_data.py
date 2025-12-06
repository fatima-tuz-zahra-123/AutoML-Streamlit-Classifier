import pandas as pd
import numpy as np

# Create a dummy dataset (simulating Titanic passenger data)
data = {
    'Age': [22, 38, 26, 35, 35, np.nan, 54, 2, 27, 14], # Has a NaN (missing value)
    'Fare': [7.25, 71.28, 7.92, 53.10, 8.05, 8.45, 51.86, 21.07, 11.13, 30.07],
    'Survived': [0, 1, 1, 1, 0, 0, 0, 0, 1, 1],
    'Class': ['Third', 'First', 'Third', 'First', 'Third', 'Third', 'First', 'Third', 'Third', 'Second'],
    'Gender': ['Male', 'Female', 'Female', 'Female', 'Male', 'Male', 'Male', 'Male', 'Female', 'Female']
}

df = pd.DataFrame(data)

# Intentionally add duplicate rows (rows 0 and 1 repeated)
df = pd.concat([df, df.iloc[[0, 1]]], ignore_index=True)

# Save to CSV
df.to_csv('test_dataset.csv', index=False)
print("Created 'test_dataset.csv' with missing values and duplicates!")
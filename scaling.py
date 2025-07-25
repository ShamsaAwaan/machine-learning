import pandas as pd
import numpy as np
df = pd.read_csv('Iris.csv')
print(df.head())

numeric_df = df.select_dtypes(include=[np.number])
max_vals = np.max(np.abs(numeric_df))
print(max_vals)

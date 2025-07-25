import pandas as pd
import numpy as np
df = pd.read_csv('Iris.csv')
print(df.head())

max_vals = np.max(np.abs(df))
print(max_vals)

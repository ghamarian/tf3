import pandas as pd
import numpy as np


df = pd.DataFrame()

data_size = 10000
df['size'] = np.random.randint(1000, 5000, size=data_size)
df['type'] = np.random.choice(['apt', 'house'], size=data_size)
df['price'] = np.random.randint(1000, 5000, size=data_size)

df.to_csv("input.csv", index=False)
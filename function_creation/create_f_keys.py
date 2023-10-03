import pandas as pd
import numpy as np

n = 50
f_keys = [np.random.randint(0, 100000) for i in range(n)]
df = pd.DataFrame(f_keys, columns=["f_keys"])
df.to_csv("function_creation/f_keys.csv", index=False)

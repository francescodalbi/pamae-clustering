import pandas as pd

df = pd.read_fwf('s1.txt', header=None)
df.to_csv('s1.csv', index=False, header=['c0','c1'])
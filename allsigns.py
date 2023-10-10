import pandas as pd

train = pd.read_csv('/Users/pavan/GIT/ISLRv2/data/train.csv')
unique_signs = train['sign'].unique()

print(unique_signs)
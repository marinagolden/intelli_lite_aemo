import pandas as pd

# Example DataFrame with MultiIndex columns
data_multi = {'Value': [1, 2, 3], 'OtherValue': [4, 5, 6]}
columns_multi = pd.MultiIndex.from_tuples([('Datetime', 'Year'), ('Data', 'Value'), ('Data', 'OtherValue')])
df_multi = pd.DataFrame(data_multi, columns=columns_multi)
print(df_multi)

# Example DataFrame with single index
data_single = {'Year': [2019, 2020, 2021], 'AnotherValue': [7, 8, 9]}
df_single = pd.DataFrame(data_single)

# Reset the index of df_multi to merge by common column 'Year'
df_multi_reset = df_multi.reset_index()

# Merge the two DataFrames on the common column 'Year'
result = pd.merge(df_multi_reset, df_single, on='Year')

print(result)

"""
NEED TO MAP PIPE SIZE TO ALL FILES FIRST.

converts partque files to csv to use in c

"""
import pandas as pd
import glob

files = glob.glob("/*.parquet")

dfs = [pd.read_parquet(file) for file in files]
df = pd.concat(dfs, ignore_index=True)
df.to_csv("combined_output.csv", index=False)

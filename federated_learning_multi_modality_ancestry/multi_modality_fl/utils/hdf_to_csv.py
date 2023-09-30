import pandas as pd
import sys
print("args", sys.argv)
fpath = sys.argv[1]
if len(sys.argv)>2:
    key = sys.argv[2]
    df = pd.read_hdf(fpath, key=key)
else:
    df = pd.read_hdf(fpath)

df.to_csv(sys.stdout, index=False)
# usage: python3 hdf2df.py data.hf > data.csv
# https://stackoverflow.com/questions/23758893/converting-hdf5-to-csv-or-tsv-files

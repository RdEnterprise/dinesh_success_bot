# inspect_csv.py
import os, sys
import pandas as pd

fn = "XAUUSD.csv"
if not os.path.exists(fn):
    print("File not found:", fn)
    sys.exit(1)

df = pd.read_csv(fn, nrows=5)
print("COLUMNS (raw):", list(df.columns))
# show stripped/lowercase version
print("COLUMNS (normalized):", [c.strip().lower() for c in df.columns])
print("\nFirst rows:")
print(df.head())

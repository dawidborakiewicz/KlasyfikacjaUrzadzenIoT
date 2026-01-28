import pandas as pd
from sqlalchemy.dialects.mssql.information_schema import columns

x = pd.read_csv("output.csv")
print(x.columns)
print(len(x.columns))
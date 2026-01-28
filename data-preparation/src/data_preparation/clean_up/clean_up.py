import pandas as pd

def handle_missing_values(df):

    df_filled = df.fillna(-1)
    missing_counts = df.isnull().sum()
    if missing_counts.sum() > 0:
        print("Zastąpiono wartości NULL w kolumnach:")
        print(missing_counts[missing_counts > 0])

    return df_filled
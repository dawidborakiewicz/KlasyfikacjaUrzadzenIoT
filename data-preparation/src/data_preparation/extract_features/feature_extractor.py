import pandas as pd


def split_columns(df: pd.DataFrame, cols_keep=None):
    if cols_keep is None:
        cols_keep = ["frame.time_epoch", "eth.src", "ip.src"]
    cols_keep = [cols_keep] if isinstance(cols_keep, str) else list(cols_keep)

    cols_keep_existing = [col for col in cols_keep if col in df.columns]

    missing_cols = set(cols_keep) - set(cols_keep_existing)
    if missing_cols:
        print(f"Warning: Pomijam brakujące kolumny: {missing_cols}")

    if not cols_keep_existing:
        print("Warning: Żadna z wymaganych kolumn nie istnieje w DataFrame")
        df_keep = pd.DataFrame()
        df_rest = df.copy()
    else:
        df_keep = df.loc[:, cols_keep_existing].copy()
        df_rest = df.drop(columns=cols_keep_existing).copy()

    return df_keep, df_rest
from clean_up.clean_up import CleanUp
import pandas as pd
from one_hot.one_hot import one_hot
from clean_up.clean_up import handle_missing_values




def main():
    df = pd.read_csv("single.csv")
    df = handle_missing_values(df)
    df = one_hot(df)
    df.to_csv("output.csv", index=False)


if __name__ == "__main__":
    main()
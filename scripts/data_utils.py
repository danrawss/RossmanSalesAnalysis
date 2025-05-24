import pandas as pd

def load_data():
    """
    Load and merge train + store datasets.
    Returns a DataFrame with a Date column parsed.
    """
    df_train = pd.read_csv("data/train.csv", parse_dates=["Date"])
    df_store = pd.read_csv("data/store.csv")
    df = df_train.merge(df_store, on="Store", how="left")
    return df

def prepare_features(df):
    """
    From the merged df, create:
      - X: numeric feature matrix
      - y: binary target HighSales
    """
    df = df.copy()
    df["HighSales"] = (df.Sales > df.Sales.median()).astype(int)
    X = df[["CompetitionDistance", "Promo2SinceWeek"]].fillna(0)
    y = df["HighSales"]
    return X, y

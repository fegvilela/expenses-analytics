import pandas as pd
import os
from glob import glob
import unicodedata

# setup your path for csv folder
DATA_PATH = "./data"


def create_agg_df(data_path: str = DATA_PATH) -> pd.DataFrame:
    # all .csv files in data_path folder
    data_files = glob(os.path.join(data_path, "*.csv"))
    print(f"\n {len(data_files)} files read from {data_path} \n")

    # reads each CSV
    dfs = []
    for file in data_files:
        df = pd.read_csv(file)
        print(f"appending file {file.split('/')[-1]} with {len(df)} rows... \n")
        dfs.append(df)

    df_agg = pd.concat(dfs, ignore_index=True)

    return df_agg


def rename_cols(df: pd.DataFrame, new_cols: list):
    print(f"renaming cols from {df.columns}... to {new_cols}")
    df.columns = new_cols


def normalize_text(text):
    """Remove accent marks, apply lowercase and remove special chars"""
    if pd.isna(text):
        return text

    text = str(text)
    text = text.lower()  # lowercase

    # Remove accent marks (NFD = Normalization Form Decomposition)
    text = unicodedata.normalize("NFD", text)
    text = "".join(char for char in text if unicodedata.category(char) != "Mn")

    # Remove special chars
    text = "".join(char for char in text if char.isalnum() or char.isspace())
    text = " ".join(text.split())  # Remove extra spaces

    return text


def extract_date(df: pd.DataFrame):
    # Converter para datetime primeiro
    df["date"] = pd.to_datetime(df["date"], format='%d/%m/%Y')

    print(df["date"].dt.year) # type: ignore
    print(df["date"].dt.month_name()) # type: ignore
    
    
    # Extrair componentes
    # df["ano"] = df["date"].dt.year
    # df["mes"] = df["data"].dt.month
    # df["dia_semana"] = df["data"].dt.day_name()
    # df["trimestre"] = df["data"].dt.quarter


def main():
    df = create_agg_df()

    # normalizations
    rename_cols(df, ["date", "value", "id", "description"])
    df["description"] = df["description"].apply(normalize_text)
    extract_date(df)
    # print(df.head(n=10))


if __name__ == "__main__":
    main()

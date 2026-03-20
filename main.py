import pandas as pd
import os
from glob import glob
import unicodedata
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

DATA_PATH = "./data"

CATEGORIES = [
    "Alimentação e Restaurantes",
    "Supermercado e Mantimentos",
    "Farmácia e Medicamentos",
    "Transporte e Combustível",
    "Lazer e Entretenimento",
    "Vestuário e Roupas",
    "Contas e Serviços (luz, água, internet)",
    "Transferências e Pix",
    "Investimentos e Aplicações",
    "Receitas e Rendimentos",
    "Beleza e Cuidados Pessoais",
    "Educação",
    "Saúde e Plano Médico",
    "Estorno ou Reembolso",
    "Outros",
]


def create_agg_df(data_path: str = DATA_PATH) -> pd.DataFrame:
    data_files = glob(os.path.join(data_path, "*.csv"))
    print(f"\n {len(data_files)} files read from {data_path} \n")

    dfs = []
    for file in data_files:
        df = pd.read_csv(file)
        print(f"appending file {file.split('/')[-1]} with {len(df)} rows... \n")
        dfs.append(df)

    df_agg = pd.concat(dfs, ignore_index=True)

    return df_agg


def rename_cols(df: pd.DataFrame, new_cols: list):
    print(f"renaming cols from {df.columns} to {new_cols}")
    df.columns = new_cols


def normalize_text(text):
    if pd.isna(text):
        return text

    text = str(text)
    text = text.lower()

    text = unicodedata.normalize("NFD", text)
    text = "".join(char for char in text if unicodedata.category(char) != "Mn")

    text = "".join(char for char in text if char.isalnum() or char.isspace())
    text = " ".join(text.split())

    return text


def extract_date(df: pd.DataFrame):
    df["date"] = pd.to_datetime(df["date"], format="%d/%m/%Y")
    print(df["date"].dt.year)
    print(df["date"].dt.month_name())


def categorize_expenses(df: pd.DataFrame, model_name: str = "paraphrase-multilingual-MiniLM-L12-v2") -> pd.DataFrame:
    print(f"\nLoading embedding model: {model_name}...")
    model = SentenceTransformer(model_name)

    print("Generating embeddings for descriptions...")
    description_embeddings = model.encode(df["description"].tolist(), show_progress_bar=True)

    print("Generating embeddings for categories...")
    category_embeddings = model.encode(CATEGORIES, show_progress_bar=True)

    print("Computing similarity and categorizing...")
    similarities = cosine_similarity(description_embeddings, category_embeddings)
    category_indices = np.argmax(similarities, axis=1)
    confidence_scores = np.max(similarities, axis=1)

    df["category"] = [CATEGORIES[i] for i in category_indices]
    df["category_confidence"] = confidence_scores

    return df


def main():
    df = create_agg_df()

    rename_cols(df, ["date", "value", "id", "description"])
    df["description"] = df["description"].apply(normalize_text)
    extract_date(df)

    df = categorize_expenses(df)

    print("\n--- Sample Results ---")
    print(df[["description", "category", "category_confidence"]].head(20))

    print("\n--- Category Distribution ---")
    print(df["category"].value_counts())

    output_path = os.path.join(DATA_PATH, "categorized_expenses.csv")
    df.to_csv(output_path, index=False)
    print(f"\nSaved categorized expenses to: {output_path}")


if __name__ == "__main__":
    main()

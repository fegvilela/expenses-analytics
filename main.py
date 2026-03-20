import pandas as pd
import os
import json
from glob import glob
import unicodedata
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

DATA_PATH = "./data"
CACHE_PATH = "./category_overrides.json"
EMBEDDINGS_CACHE = "./embeddings_cache.json"

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


def load_overrides() -> dict:
    if os.path.exists(CACHE_PATH):
        with open(CACHE_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_overrides(overrides: dict):
    with open(CACHE_PATH, "w", encoding="utf-8") as f:
        json.dump(overrides, f, ensure_ascii=False, indent=2)


def add_override(description: str, category: str):
    overrides = load_overrides()
    overrides[description] = category
    save_overrides(overrides)


def remove_override(description: str):
    overrides = load_overrides()
    if description in overrides:
        del overrides[description]
        save_overrides(overrides)


def list_overrides():
    overrides = load_overrides()
    if not overrides:
        print("No overrides saved.")
        return
    for desc, cat in overrides.items():
        print(f"  '{desc}' -> '{cat}'")


def create_agg_df(data_path: str = DATA_PATH) -> pd.DataFrame:
    data_files = glob(os.path.join(data_path, "*.csv"))
    print(f"\n {len(data_files)} files read from {data_path}")

    dfs = []
    for file in data_files:
        df = pd.read_csv(file)
        if "category" in df.columns:
            print(f"skipping already categorized file {file.split('/')[-1]}...")
            continue
        print(f"appending file {file.split('/')[-1]} with {len(df)} rows...")
        dfs.append(df)

    return pd.concat(dfs, ignore_index=True)


def rename_cols(df: pd.DataFrame, new_cols: list):
    df.columns = new_cols


def normalize_text(text):
    if pd.isna(text):
        return ""
    text = str(text)
    text = text.lower()
    text = unicodedata.normalize("NFD", text)
    text = "".join(char for char in text if unicodedata.category(char) != "Mn")
    text = "".join(char for char in text if char.isalnum() or char.isspace())
    return " ".join(text.split())


def categorize_expenses(df: pd.DataFrame) -> pd.DataFrame:
    overrides = load_overrides()
    model_name = "paraphrase-multilingual-MiniLM-L12-v2"

    print(f"\nLoading model: {model_name}...")
    model = SentenceTransformer(model_name)

    if os.path.exists(EMBEDDINGS_CACHE):
        with open(EMBEDDINGS_CACHE, "r", encoding="utf-8") as f:
            embeddings_cache = json.load(f)
    else:
        embeddings_cache = {}

    print("Generating embeddings...")
    descriptions_normalized = [normalize_text(d) for d in df["description"]]

    for norm in set(descriptions_normalized):
        if norm not in embeddings_cache:
            emb = model.encode([norm])[0].tolist()
            embeddings_cache[norm] = emb

    with open(EMBEDDINGS_CACHE, "w", encoding="utf-8") as f:
        json.dump(embeddings_cache, f, ensure_ascii=False)

    desc_embeddings = np.array([embeddings_cache[d] for d in descriptions_normalized])

    print("Computing categories...")
    category_embeddings = model.encode(CATEGORIES)
    similarities = cosine_similarity(desc_embeddings, category_embeddings)
    category_indices = np.argmax(similarities, axis=1)
    confidence_scores = np.max(similarities, axis=1)

    df["category"] = [CATEGORIES[i] for i in category_indices]
    df["category_confidence"] = confidence_scores
    df["description_normalized"] = descriptions_normalized

    for idx, norm_desc in enumerate(df["description_normalized"]):
        if norm_desc in overrides:
            df.at[idx, "category"] = overrides[norm_desc]
            df.at[idx, "category_confidence"] = 1.0

    return df


def review_categories(df: pd.DataFrame):
    print("\n=== REVIEW CATEGORIES ===")
    print("\nCategories available:")
    for i, cat in enumerate(CATEGORIES):
        print(f"  [{i}] {cat}")
    print(f"  [c] Correct multiple")
    print(f"  [q] Quit\n")

    while True:
        print("\n--- Options ---")
        print("1. View low confidence (< 0.5)")
        print("2. View by category")
        print("3. Correct a description")
        print("4. Correct by partial match")
        print("5. View/Clear overrides")
        print("q. Quit and save")

        choice = input("\nChoice: ").strip()

        if choice == "q":
            break

        elif choice == "1":
            low_conf = df[df["category_confidence"] < 0.5][["description", "category", "category_confidence"]]
            print(f"\n{len(low_conf)} items with low confidence:")
            print(low_conf.to_string())

        elif choice == "2":
            cat = input("Category name or number: ").strip()
            if cat.isdigit():
                cat = CATEGORIES[int(cat)]
            filtered = df[df["category"] == cat][["description", "category_confidence"]]
            print(f"\n{len(filtered)} items in '{cat}':")
            print(filtered.to_string())

        elif choice == "3":
            desc = input("Exact description: ").strip()
            mask = df["description_normalized"] == normalize_text(desc)
            if not mask.any():
                print("Description not found.")
                continue
            current = df.loc[mask, "category"].values[0]
            print(f"Current category: {current}")
            new_cat = input("New category (name or number): ").strip()
            if new_cat.isdigit():
                new_cat = CATEGORIES[int(new_cat)]
            add_override(normalize_text(desc), new_cat)
            df.loc[mask, "category"] = new_cat
            df.loc[mask, "category_confidence"] = 1.0
            print(f"Updated: '{desc}' -> '{new_cat}'")

        elif choice == "4":
            partial = input("Partial text to match: ").strip().lower()
            mask = df["description_normalized"].str.contains(partial, regex=False)
            if not mask.any():
                print("No matches found.")
                continue
            print(f"\n{mask.sum()} matches found:")
            print(df.loc[mask, ["description", "category"]].to_string())
            new_cat = input("New category for all matches (name or number): ").strip()
            if new_cat.isdigit():
                new_cat = CATEGORIES[int(new_cat)]
            for idx in df[mask].index:
                norm = df.loc[idx, "description_normalized"]
                add_override(norm, new_cat)
                df.loc[idx, "category"] = new_cat
                df.loc[idx, "category_confidence"] = 1.0
            print(f"Updated {mask.sum()} items to '{new_cat}'")

        elif choice == "5":
            print("\nCurrent overrides:")
            list_overrides()
            if input("\nClear all overrides? (y/N): ").strip().lower() == "y":
                save_overrides({})
                print("Overrides cleared.")

    return df


def main():
    import sys

    if len(sys.argv) > 1:
        if sys.argv[1] == "review":
            df = create_agg_df()
            rename_cols(df, ["date", "value", "id", "description"])
            df = categorize_expenses(df)
            df = review_categories(df)
            output = os.path.join(DATA_PATH, "categorized_expenses.csv")
            df.to_csv(output, index=False)
            print(f"\nSaved to {output}")
        elif sys.argv[1] == "overrides":
            list_overrides()
        elif sys.argv[1] == "clear":
            save_overrides({})
            print("Overrides cleared.")
        return

    df = create_agg_df()
    rename_cols(df, ["date", "value", "id", "description"])
    df["date_parsed"] = pd.to_datetime(df["date"], format="%d/%m/%Y")
    df = df.dropna(subset=["description"])
    df = df[df["description"].str.strip() != ""]
    df = categorize_expenses(df)

    print("\n--- Sample Results ---")
    print(df[["description", "category", "category_confidence"]].head(15))

    print("\n--- Category Distribution ---")
    print(df["category"].value_counts())

    output = os.path.join(DATA_PATH, "categorized_expenses.csv")
    df.to_csv(output, index=False)
    print(f"\nSaved to {output}")


if __name__ == "__main__":
    main()

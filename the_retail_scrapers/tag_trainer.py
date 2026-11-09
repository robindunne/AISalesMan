#!/usr/bin/env python3
"""
train_tag_models.py

Train multi-label classifiers for IKEA product style_tags and persona_tags
using TF-IDF + One-vs-Rest Logistic Regression, and optionally update Supabase.

Usage (training only):
    python train_tag_models.py --csv products_training_sample_200_with_style_tags.csv

Usage (after verifying models, to update Supabase tags):
    python train_tag_models.py --update-supabase
    (Make sure .env has SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY)
"""

import argparse
import os
from typing import List, Tuple

import joblib
import pandas as pd
from dotenv import load_dotenv
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MultiLabelBinarizer


# ------------------------
# Data loading and preprocessing
# ------------------------

def load_labeled_data(csv_path: str) -> Tuple[pd.DataFrame, List[str], List[List[str]], List[List[str]]]:
    """
    Load labeled training data from the CSV and return:
      - df: original dataframe
      - product_texts: raw_description per row
      - style_labels: list of label lists from style_tags
      - persona_labels: list of label lists from persona_tags
    """
    df = pd.read_csv(csv_path)

    # Text input = description only
    product_texts = df["raw_description"].fillna("").tolist()

    def split_tags(col: pd.Series) -> List[List[str]]:
        labels = []
        for v in col:
            if isinstance(v, str) and v.strip():
                labels.append(v.split(","))
            else:
                labels.append([])
        return labels

    style_labels = split_tags(df["style_tags"])
    persona_labels = split_tags(df["persona_tags"])

    return df, product_texts, style_labels, persona_labels


# ------------------------
# Generic training function
# ------------------------

def train_multilabel_model(
    product_texts: List[str],
    product_labels: List[List[str]],
    prefix: str,
    max_features: int = 5000,
) -> None:
    """
    Train a multi-label classifier and save:
      - {prefix}_vectorizer.joblib
      - {prefix}_label_binarizer.joblib
      - {prefix}_classifier.joblib
    """
    print(f"\n=== Training {prefix} model ===")

    # 1) Vectorize text
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        ngram_range=(1, 2),
        stop_words="english",
    )
    X = vectorizer.fit_transform(product_texts)

    # 2) Encode labels
    mlb = MultiLabelBinarizer()
    Y = mlb.fit_transform(product_labels)

    # 3) Train and validation split
    X_train, X_val, Y_train, Y_val = train_test_split(
        X, Y, test_size=0.2, random_state=42
    )

    clf = OneVsRestClassifier(
        LogisticRegression(
            max_iter=500,
            C=2.0,
            class_weight="balanced",
            n_jobs=-1,
        )
    )

    clf.fit(X_train, Y_train)

    # 4) Evaluate
    Y_pred = clf.predict(X_val)
    micro = f1_score(Y_val, Y_pred, average="micro", zero_division=0)
    macro = f1_score(Y_val, Y_pred, average="macro", zero_division=0)

    print(f"{prefix} labels: {list(mlb.classes_)}")
    print(f"{prefix} Micro F1: {micro:.3f}")
    print(f"{prefix} Macro F1: {macro:.3f}")

    # 5) Save everything
    vec_path = f"{prefix}_vectorizer.joblib"
    mlb_path = f"{prefix}_label_binarizer.joblib"
    clf_path = f"{prefix}_classifier.joblib"

    joblib.dump(vectorizer, vec_path)
    joblib.dump(mlb, mlb_path)
    joblib.dump(clf, clf_path)

    print(f"Saved {prefix} vectorizer to {vec_path}")
    print(f"Saved {prefix} label binarizer to {mlb_path}")
    print(f"Saved {prefix} classifier to {clf_path}")


# ------------------------
# Supabase integration (optional, run after you are happy with models)
# ------------------------

def apply_models_to_supabase(
    supabase_url_env: str = "SUPABASE_URL",
    supabase_key_env: str = "SUPABASE_SERVICE_ROLE_KEY",
    table_name: str = "products",
    batch_size: int = 100,
) -> None:
    """
    Load trained models, connect to Supabase, and fill style_tags and persona_tags
    for products that do not have them yet.

    Requirements:
      pip install supabase-py python-dotenv

    Make sure .env has:
      SUPABASE_URL=...
      SUPABASE_SERVICE_ROLE_KEY=...
    """
    from supabase import create_client, Client

    # Env already loaded by load_dotenv() in main
    url = os.environ.get(supabase_url_env)
    key = os.environ.get(supabase_key_env)

    if not url or not key:
        raise RuntimeError(
            f"Missing Supabase credentials. Set {supabase_url_env} and {supabase_key_env} in your .env or environment."
        )

    supabase: Client = create_client(url, key)

    # Load models
    style_vectorizer = joblib.load("style_vectorizer.joblib")
    style_mlb = joblib.load("style_label_binarizer.joblib")
    style_clf = joblib.load("style_classifier.joblib")

    persona_vectorizer = joblib.load("persona_vectorizer.joblib")
    persona_mlb = joblib.load("persona_label_binarizer.joblib")
    persona_clf = joblib.load("persona_classifier.joblib")

    print("\n=== Applying models to Supabase products ===")

    offset = 0
    total_updated = 0

    while True:
        # Fetch a page of products
        resp = (
            supabase.table(table_name)
            .select("*")
            .range(offset, offset + batch_size - 1)
            .execute()
        )

        rows = resp.data or []
        if not rows:
            break

        for row in rows:
            # Skip rows that already have style_tags (tweak if you want to always overwrite)
            style_existing = row.get("style_tags")
            persona_existing = row.get("persona_tags")

            if style_existing and str(style_existing).strip():
                continue

            # Use description only for prediction
            text = row.get("raw_description", "") or ""
            if not text.strip():
                continue

            X_style = style_vectorizer.transform([text])
            X_persona = persona_vectorizer.transform([text])

            y_style = style_clf.predict(X_style)
            y_persona = persona_clf.predict(X_persona)

            style_tags = list(style_mlb.inverse_transform(y_style)[0])      # tuple -> list
            persona_tags = list(persona_mlb.inverse_transform(y_persona)[0])

            # Optional: if both are empty, skip updating this row
            if not style_tags and not persona_tags:
                continue

            supabase.table(table_name).update({
                "style_tags": style_tags,          # Python list -> Postgres text[]
                "persona_tags": persona_tags,
            }).eq("id", row["id"]).execute()

            total_updated += 1

        offset += batch_size
        print(f"Processed {offset} rows... (updated so far: {total_updated})")

    print(f"\nDone. Total rows updated: {total_updated}")


# ------------------------
# Main CLI
# ------------------------

def main():
    # Load env from .env at startup
    load_dotenv()

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--csv",
        type=str,
        default="products_training_sample_200_with_style_tags.csv",
        help="Path to training CSV",
    )
    parser.add_argument(
        "--update-supabase",
        action="store_true",
        help="After training and verification, apply trained models to Supabase products.",
    )
    args = parser.parse_args()

    print("Loading labeled data from:", args.csv)
    df, product_texts, style_labels, persona_labels = load_labeled_data(args.csv)

    # Train style model
    train_multilabel_model(
        product_texts=product_texts,
        product_labels=style_labels,
        prefix="style",
    )

    # Train persona model
    train_multilabel_model(
        product_texts=product_texts,
        product_labels=persona_labels,
        prefix="persona",
    )

    # Optional: update Supabase
    if args.update_supabase:
        apply_models_to_supabase()


if __name__ == "__main__":
    main()

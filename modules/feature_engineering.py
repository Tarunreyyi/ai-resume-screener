# modules/feature_engineering.py

import os
import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler


# -----------------------------
# EDUCATION ENCODING MAP
# -----------------------------
EDUCATION_MAP = {
    "phd": 4,
    "mba": 3,
    "m.tech": 3,
    "mtech": 3,
    "b.tech": 2,
    "btech": 2,
    "b.sc": 2,
    "bsc": 2
}


# -----------------------------
# TF-IDF FEATURE GENERATION
# -----------------------------
def generate_skill_features(df: pd.DataFrame, model_path: str):
    """
    Convert Skills column into TF-IDF vectors
    Save trained vectorizer
    """

    vectorizer = TfidfVectorizer()

    skill_matrix = vectorizer.fit_transform(df["Skills"])

    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(vectorizer, model_path)

    return skill_matrix, vectorizer


# -----------------------------
# EDUCATION ENCODING
# -----------------------------
def encode_education(df: pd.DataFrame):
    df["Education_Score"] = (
        df["Education"]
        .astype(str)
        .str.lower()
        .map(EDUCATION_MAP)
        .fillna(1)
    )
    return df


# -----------------------------
# NORMALIZE NUMERIC FEATURES
# -----------------------------
def normalize_features(df: pd.DataFrame):
    scaler = MinMaxScaler()

    df[["Experience_Score", "Projects_Score"]] = scaler.fit_transform(
        df[["Experience (Years)", "Projects Count"]]
    )

    return df


# -----------------------------
# CERTIFICATION ENCODING
# -----------------------------
def encode_certification(df: pd.DataFrame):
    df["Certification_Score"] = df["Certifications"].apply(
        lambda x: 1 if str(x).strip() != "" else 0
    )
    return df


# -----------------------------
# COMPLETE FEATURE ENGINEERING
# -----------------------------
def build_features(df: pd.DataFrame, model_path: str):
    """
    Complete feature engineering pipeline
    """

    # TF-IDF
    skill_matrix, vectorizer = generate_skill_features(df, model_path)

    # Education
    df = encode_education(df)

    # Normalize experience + projects
    df = normalize_features(df)

    # Certification
    df = encode_certification(df)

    return df, skill_matrix

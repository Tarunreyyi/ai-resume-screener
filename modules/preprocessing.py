# modules/preprocessing.py

import os
import pandas as pd


def clean_skills(text: str) -> str:
    """
    Clean skills text:
    - Lowercase
    - Remove extra spaces
    - Standardize comma separation
    """
    if pd.isna(text):
        return ""

    skills = [skill.strip().lower() for skill in str(text).split(",")]
    return ",".join(skills)


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Perform full preprocessing on resume dataset.
    """

    df = df.copy()

    # -----------------------------
    # Clean Skills
    # -----------------------------
    df["Skills"] = df["Skills"].apply(clean_skills)

    # -----------------------------
    # Convert Experience to numeric
    # -----------------------------
    df["Experience (Years)"] = pd.to_numeric(
        df["Experience (Years)"],
        errors="coerce"
    ).fillna(0)

    # -----------------------------
    # Convert Projects Count to numeric
    # -----------------------------
    df["Projects Count"] = pd.to_numeric(
        df["Projects Count"],
        errors="coerce"
    ).fillna(0)

    # -----------------------------
    # Clean Certifications
    # -----------------------------
    df["Certifications"] = df["Certifications"].fillna("").astype(str).str.strip()

    return df


def save_cleaned_data(df: pd.DataFrame, output_path: str):
    """
    Save cleaned dataset to processed folder.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)

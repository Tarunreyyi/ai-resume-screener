# modules/scoring.py

import os
import pandas as pd
from utils.helper import WEIGHTS


def calculate_final_scores(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate weighted final score (0–100 scale).
    """

    df = df.copy()

    # Normalize education (1–4 → 0–1)
    df["Education_Normalized"] = df["Education_Score"] / 4

    # Weighted score (0–1)
    df["Final_Score"] = (
        WEIGHTS["skill"] * df["Skill_Score"] +
        WEIGHTS["experience"] * df["Experience_Score"] +
        WEIGHTS["education"] * df["Education_Normalized"] +
        WEIGHTS["projects"] * df["Projects_Score"] +
        WEIGHTS["certification"] * df["Certification_Score"]
    )

    # Convert to 0–100 scale
    df["Final_Score"] = (df["Final_Score"] * 100).round(2)

    return df


def save_final_scores(df: pd.DataFrame, output_path: str):
    """
    Save final scored dataset.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)

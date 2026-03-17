# modules/ranking.py

import os
import pandas as pd


def rank_candidates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Sort candidates by Final_Score (descending)
    and assign ranking.
    """

    df = df.copy()

    # Sort by highest score first
    df = df.sort_values(
        by="Final_Score",
        ascending=False
    ).reset_index(drop=True)

    # Assign Rank
    df["Rank"] = range(1, len(df) + 1)

    return df


def save_ranked_candidates(df: pd.DataFrame, output_path: str):
    """
    Save ranked candidates to CSV.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)

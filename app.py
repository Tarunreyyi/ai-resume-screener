# app.py

import streamlit as st
import pandas as pd

from modules.data_loader import load_dataset
from modules.preprocessing import clean_data
from modules.feature_engineering import build_features
from modules.similarity import calculate_skill_similarity
from modules.scoring import calculate_final_scores
from modules.ranking import rank_candidates
from modules.resume_parser import extract_text_from_pdf


# -------------------------------
# CONFIG
# -------------------------------
DATA_PATH = "data/raw/ai_resume_dataset_2025.csv"
MODEL_PATH = "models/tfidf_vectorizer.pkl"


# -------------------------------
# MAIN APP
# -------------------------------
def main():

    st.set_page_config(page_title="AI Resume Screener", layout="wide")
    st.title("🤖 AI Resume Screener")

    # ---------------- LOAD DATASET ----------------
    df = load_dataset(DATA_PATH)
    st.success("Dataset Loaded Successfully")

    # ---------------- FILTER ROLE ----------------
    st.sidebar.header("Filter Options")
    job_roles = sorted(df["Job Role"].dropna().unique())

    selected_role = st.sidebar.selectbox(
        "Select Job Role",
        job_roles
    )

    filtered_df = df[df["Job Role"] == selected_role].copy()

    # ---------------- CLEAN DATA ----------------
    cleaned_df = clean_data(filtered_df)

    # ---------------- FEATURE ENGINEERING ----------------
    featured_df, skill_matrix = build_features(
        cleaned_df,
        MODEL_PATH
    )

    # ---------------- SKILL SIMILARITY ----------------
    skill_scores = calculate_skill_similarity(
        skill_matrix,
        selected_role,
        MODEL_PATH
    )

    featured_df["Skill_Score"] = skill_scores

    # ---------------- FINAL SCORE ----------------
    scored_df = calculate_final_scores(featured_df)

    # ---------------- RANKING ----------------
    ranked_df = rank_candidates(scored_df)

    # ---------------- DISPLAY ----------------
    st.subheader(f"Ranked Candidates for: {selected_role}")

    display_cols = [
        "Name",
        "Skills",
        "Experience (Years)",
        "Final_Score",
        "Rank",
        "AI Score"
    ]

    available_cols = [c for c in display_cols if c in ranked_df.columns]
    final_display_df = ranked_df[available_cols]

    st.dataframe(final_display_df, use_container_width=True)

    # ---------------- DOWNLOAD CSV ----------------
    csv_data = final_display_df.to_csv(index=False).encode("utf-8")

    st.download_button(
        label="⬇ Download Ranked Candidates (CSV)",
        data=csv_data,
        file_name=f"ranked_candidates_{selected_role}.csv",
        mime="text/csv"
    )

    # ====================================================
    #               RESUME UPLOAD SECTION
    # ====================================================
    st.markdown("---")
    st.subheader("📄 Upload Resume (PDF)")

    uploaded_file = st.file_uploader(
        "Upload resume to check match percentage",
        type=["pdf"]
    )

    if uploaded_file is not None:

        resume_text = extract_text_from_pdf(uploaded_file)
        resume_text_lower = resume_text.lower()

        # Create temporary DF
        resume_df = pd.DataFrame({
            "Skills": [resume_text],
            "Experience (Years)": [5],
            "Projects Count": [5],
            "Education": ["PhD"],
            "Certifications": ["Yes"]
        })

        resume_df = clean_data(resume_df)

        resume_features, resume_matrix = build_features(
            resume_df,
            MODEL_PATH
        )

        # Base similarity
        skill_score = calculate_skill_similarity(
            resume_matrix,
            selected_role,
            MODEL_PATH
        )[0]

        # ---- Skill Bonus Boosting ----
        important_keywords = [
            "machine learning",
            "deep learning",
            "nlp",
            "tensorflow",
            "pytorch",
            "python",
            "transformers",
            "neural networks"
        ]

        bonus = 0
        for word in important_keywords:
            if word in resume_text_lower:
                bonus += 0.05

        final_score = min(skill_score + bonus, 1.0)

        st.success(f"Resume Match Score: {round(final_score * 100, 2)}%")

        st.info("Score based on skill similarity + keyword strength boost.")


if __name__ == "__main__":
    main()
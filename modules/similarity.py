import joblib
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


# Role → Skill mapping (VERY IMPORTANT FIX)
ROLE_SKILLS = {
    "AI Researcher": "machine learning deep learning nlp tensorflow pytorch python transformers neural networks",
    "Data Scientist": "python sql machine learning data analysis pandas numpy statistics",
    "Software Engineer": "java c++ react sql data structures algorithms",
    "Cybersecurity Analyst": "cybersecurity networking ethical hacking linux security",
}


def calculate_skill_similarity(skill_matrix, selected_role, model_path):
    """
    Compare candidate skills with role skill keywords
    """

    vectorizer = joblib.load(model_path)

    # Get role skill text
    role_text = ROLE_SKILLS.get(selected_role, selected_role.lower())

    # Convert role skills to vector
    role_vector = vectorizer.transform([role_text])

    # Compute cosine similarity
    similarities = cosine_similarity(skill_matrix, role_vector)

    # Flatten to 1D array
    return similarities.flatten()
# 🤖 AI Resume Screener

An AI-powered resume screening system that automates candidate evaluation and ranking based on job role requirements.

---

## 🚀 Features

- 📊 Candidate ranking based on multiple factors
- 🧠 Skill matching using TF-IDF + Cosine Similarity
- ⚖️ Weighted scoring system (Skills, Experience, Education, Projects)
- 📄 Resume PDF upload & evaluation
- 📈 Match percentage calculation
- 📥 Download ranked candidates as CSV
- 🎯 Real-time resume matching

---

## 🏗️ Project Pipeline

1. Load Dataset  
2. Filter by Job Role  
3. Clean Data  
4. Feature Engineering  
5. Skill Similarity Calculation  
6. Final Score Calculation  
7. Ranking Candidates  
8. Display Results  

---

## 🛠️ Tech Stack

- Python
- Streamlit
- Pandas
- Scikit-learn
- NumPy
- TF-IDF Vectorization
- Cosine Similarity

---

## 📂 Project Structure
ai_resume_screener/
|
├── app.py
├── requirements.txt
├── README.md
|
├── data/
|   ├── raw/
|   |   └── ai_resume_dataset_2025.csv
|   ├── processed/
|   |   └── cleaned_resume_data.csv
|   └── external/
|       └── job_description.txt
|
├── modules/
|   ├── data_loader.py
|   ├── preprocessing.py
|   ├── feature_engineering.py
|   ├── similarity.py
|   ├── scoring.py
|   ├── ranking.py
|   └── resume_parser.py
|
├── models/
|   └── tfidf_vectorizer.pkl
|
├── outputs/
|   ├── final_scores.csv
|   └── ranked_candidates.csv
|
└── utils/
    ├── config.py
    └── helper.py
import re
import pandas as pd
import streamlit as st
import pdfplumber
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.metrics.pairwise import cosine_similarity

EXTRA_STOP = {
    "experience", "team", "ability", "work", "working", "strong", "skills",
    "knowledge", "years", "year", "role", "responsibilities", "requirements",
    "candidate", "candidates", "including", "good", "excellent", "looking",
    "join", "company", "job", "position", "opportunity", "preferred",
    "required", "plus", "understanding", "etc", "will", "ensure", "using",
}
STOP_WORDS = list(ENGLISH_STOP_WORDS.union(EXTRA_STOP))

SYNONYMS = {
    "ml": "machine learning",
    "dl": "deep learning",
    "nlp": "natural language processing",
    "cv": "computer vision",
    "oop": "object oriented programming",
    "oops": "object oriented programming",
    "sklearn": "scikit-learn",
    "scikit learn": "scikit-learn",
    "js": "javascript",
    "postgres": "postgresql",
    "k8s": "kubernetes",
    "gen ai": "generative ai",
    "genai": "generative ai",
    "large language models": "llm",
    "rest apis": "rest api",
    "restful": "rest api",
}

SKILLS = [
    "python", "java", "javascript", "typescript", "c++", "c#", "sql", "mysql",
    "postgresql", "mongodb", "html", "css", "react", "node.js", "django",
    "flask", "fastapi", "streamlit", "git", "github", "docker", "kubernetes",
    "aws", "azure", "gcp", "linux", "rest api", "ci/cd", "flutter", "firebase",
    "android", "machine learning", "deep learning", "natural language processing",
    "computer vision", "generative ai", "llm", "transformers", "scikit-learn",
    "pandas", "numpy", "matplotlib", "seaborn", "tensorflow", "pytorch", "keras",
    "opencv", "data structures", "algorithms", "object oriented programming",
    "statistics", "data analysis", "data visualization", "feature engineering",
    "model deployment", "mlops", "power bi", "tableau", "excel", "spark",
    "hadoop", "airflow", "nlp", "langchain", "hugging face", "jupyter",
]

def normalize(text):
    text = text.lower()
    for short, full in SYNONYMS.items():
        text = re.sub(r"(?<![a-z0-9+#.\-])" + re.escape(short) + r"(?![a-z0-9+#\-])", full, text)
    return text

def has_skill(text, skill):
    pattern = r"(?<![a-z0-9+#.\-])" + re.escape(skill) + r"(?![a-z0-9+#\-])"
    return re.search(pattern, text) is not None

def read_pdf(file):
    text = ""
    with pdfplumber.open(file) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + " "
    return text

def analyze(resume_raw, jd_raw):
    resume_text = normalize(resume_raw)
    jd_clean = normalize(jd_raw)

    # 1) Skill match (main score)
    jd_skills = [s for s in SKILLS if has_skill(jd_clean, s)]
    have = [s for s in jd_skills if has_skill(resume_text, s)]
    lack = [s for s in jd_skills if s not in have]
    skill_score = (len(have) / len(jd_skills) * 100) if jd_skills else 0

    # 2) Text similarity (secondary score, single words only)
    vectorizer = TfidfVectorizer(
        stop_words=STOP_WORDS,
        token_pattern=r"[A-Za-z][A-Za-z0-9+#\-]*",
    )
    matrix = vectorizer.fit_transform([resume_text, jd_clean])
    text_score = cosine_similarity(matrix[0], matrix[1])[0][0] * 100

    # 3) Other important JD words not in the resume
    words = vectorizer.get_feature_names_out()
    resume_scores = matrix[0].toarray()[0]
    jd_scores = matrix[1].toarray()[0]
    other = sorted(
        [(j, w) for w, r, j in zip(words, resume_scores, jd_scores) if j > 0 and r == 0],
        reverse=True,
    )

    return {
        "skill_score": skill_score,
        "text_score": text_score,
        "jd_skill_count": len(jd_skills),
        "have": have,
        "lack": lack,
        "other": [w for _, w in other if w not in jd_skills],
    }

# ---------- Page ----------
st.title("Resume vs Job Description Matcher")

pdf = st.file_uploader("Upload your resume (PDF)", type="pdf")
company = st.text_input("Company name")
jd_text = st.text_area("Paste the job description here", height=250)

if st.button("Analyze"):
    if not pdf or not company.strip() or not jd_text.strip():
        st.warning("Please upload your resume, enter the company name and paste the job description.")
    else:
        result = analyze(read_pdf(pdf), jd_text)
        result["company"] = company.strip()
        st.session_state["latest"] = result

        history = [
            h for h in st.session_state.get("history", [])
            if h["Company"].lower() != company.strip().lower()
        ]
        history.append({
            "Company": company.strip(),
            "Skill match %": round(result["skill_score"]),
            "Skills asked": result["jd_skill_count"],
            "Text similarity %": round(result["text_score"]),
            "Missing skills": ", ".join(result["lack"][:10]),
        })
        st.session_state["history"] = history

if "latest" in st.session_state:
    r = st.session_state["latest"]
    st.header(f"Result for {r['company']}")
    c1, c2 = st.columns(2)
    c1.metric("Skill match", f"{r['skill_score']:.0f}%",
              help="Share of the tech skills named in the JD that appear in your resume")
    c2.metric("Text similarity", f"{r['text_score']:.0f}%",
              help="Overall word overlap. Usually low, so use it only to compare")

    if r["jd_skill_count"] == 0:
        st.info("No known tech skills were found in this JD. Check that you pasted the full description.")
    else:
        st.write(f"The JD mentions **{r['jd_skill_count']}** known skills. You have **{len(r['have'])}** of them.")

    st.subheader("Skills you have")
    st.write(", ".join(r["have"]) or "None found")
    st.subheader("Skills missing from your resume")
    st.write(", ".join(r["lack"]) or "None. You cover every skill listed.")
    st.subheader("Other important JD words not in your resume")
    st.write(", ".join(r["other"][:20]))

st.divider()
st.subheader("Saved results (this session only)")

if st.button("Clear saved results"):
    st.session_state["history"] = []
    st.session_state.pop("latest", None)

history = st.session_state.get("history", [])
if history:
    st.dataframe(pd.DataFrame(history), use_container_width=True)
else:
    st.info("No saved results yet. Analyze a job description to see it here.")
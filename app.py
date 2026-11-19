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
    "sklearn": "scikit-learn",
    "scikit learn": "scikit-learn",
    "js": "javascript",
}

def normalize(text):
    text = text.lower()
    for short, full in SYNONYMS.items():
        text = re.sub(r"\b" + re.escape(short) + r"\b", full, text)
    return text

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

    vectorizer = TfidfVectorizer(
        stop_words=STOP_WORDS,
        token_pattern=r"[A-Za-z][A-Za-z0-9+#\-]*",
        ngram_range=(1, 2),
    )
    matrix = vectorizer.fit_transform([resume_text, jd_clean])
    score = cosine_similarity(matrix[0], matrix[1])[0][0] * 100

    words = vectorizer.get_feature_names_out()
    resume_scores = matrix[0].toarray()[0]
    jd_scores = matrix[1].toarray()[0]

    missing, matched = [], []
    for word, r, j in zip(words, resume_scores, jd_scores):
        if j > 0 and r == 0:
            missing.append((j, word))
        elif j > 0 and r > 0:
            matched.append((j, word))
    missing.sort(reverse=True)
    matched.sort(reverse=True)

    total = len(matched) + len(missing)
    coverage = (len(matched) / total * 100) if total else 0
    return score, coverage, [w for _, w in missing], [w for _, w in matched]

# ---------- Page ----------
st.title("Resume vs Job Description Matcher")

pdf = st.file_uploader("Upload your resume (PDF)", type="pdf")
company = st.text_input("Company name")
jd_text = st.text_area("Paste the job description here", height=250)

if st.button("Analyze"):
    if not pdf or not company.strip() or not jd_text.strip():
        st.warning("Please upload your resume, enter the company name and paste the job description.")
    else:
        score, coverage, missing, matched = analyze(read_pdf(pdf), jd_text)

        # keep the latest full result so it stays on screen after re-runs
        st.session_state["latest"] = {
            "company": company.strip(),
            "score": score,
            "coverage": coverage,
            "missing": missing,
            "matched": matched,
        }

        # save a short row; replace an older row for the same company
        history = [
            h for h in st.session_state.get("history", [])
            if h["Company"].lower() != company.strip().lower()
        ]
        history.append({
            "Company": company.strip(),
            "Match score %": round(score),
            "Coverage %": round(coverage),
            "Top missing keywords": ", ".join(missing[:10]),
        })
        st.session_state["history"] = history

# ---------- Latest result ----------
if "latest" in st.session_state:
    r = st.session_state["latest"]
    st.header(f"Result for {r['company']}")
    col1, col2 = st.columns(2)
    col1.metric("Match score", f"{r['score']:.0f}%")
    col2.metric("Keyword coverage", f"{r['coverage']:.0f}%")

    st.subheader("Missing keywords (in the JD, not in your resume)")
    st.write(", ".join(r["missing"][:25]))
    st.subheader("Matched keywords")
    st.write(", ".join(r["matched"][:25]))

# ---------- Saved results (one section) ----------
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
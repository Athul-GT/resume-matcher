# Resume vs Job Description Matcher

A Streamlit web app that compares a PDF resume with a job description and
shows a match score and the keywords the resume is missing.

**Live demo:** https://athul-resume-matcher.streamlit.app/

![Screenshot](screenshot.png)

## Features
- Upload a PDF resume and paste a job description
- Match score and keyword coverage percentage
- Lists of matched and missing keywords
- Synonym handling (for example, ML = machine learning)
- Saved results table to compare several companies in one session

## How it works
1. pdfplumber extracts the text from the resume PDF.
2. Both texts are cleaned, and short forms like "ML" are expanded.
3. TF-IDF turns each text into numbers. Distinctive words get higher weight.
4. Cosine similarity compares the two and gives the match score.
5. Words that score above zero in the job description but are absent from
   the resume are shown as missing keywords.

## Tech stack
Python, Streamlit, scikit-learn, pdfplumber, pandas

## Run locally
    pip install -r requirements.txt
    streamlit run app.py

## Limitations
- It matches words, not meaning, so the score is a rough guide.
- Scanned or image-only PDFs have no extractable text.
- Results are stored only for the current browser session.

## Author
Athul Geo Thomas
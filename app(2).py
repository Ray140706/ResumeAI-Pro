import streamlit as st
import PyPDF2
import nltk
import string

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.corpus import stopwords

# -----------------------------
# NLTK Setup (for deployment)
# -----------------------------
try:
    stopwords.words('english')
except LookupError:
    nltk.download('stopwords', quiet=True)

# -----------------------------
# PDF TEXT EXTRACTION
# -----------------------------
def extract_text_from_pdf(uploaded_file):
    text = ""
    try:
        pdf_reader = PyPDF2.PdfReader(uploaded_file)
        for page in pdf_reader.pages:
            extracted = page.extract_text()
            if extracted:
                text += extracted
    except:
        return ""
    return text

# -----------------------------
# TEXT PREPROCESSING
# -----------------------------
def preprocess(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    words = text.split()
    stop_words = set(stopwords.words('english'))
    words = [w for w in words if w not in stop_words]
    return " ".join(words)

# -----------------------------
# KEYWORD EXTRACTION (GENERIC)
# -----------------------------
def extract_keywords(text):
    vectorizer = TfidfVectorizer(
        stop_words='english',
        max_features=100
    )
    vectorizer.fit([text])
    return set(vectorizer.get_feature_names_out())

# -----------------------------
# FILTER NON-SKILL WORDS
# -----------------------------
BAD_WORDS = {
    "using", "based", "ability", "work", "team", "good",
    "strong", "skills", "knowledge", "experience",
    "responsibilities", "required", "preferred",
    "including", "various", "related", "field",
    "understanding", "basic", "role", "job", "candidate"
}

# -----------------------------
# STREAMLIT UI
# -----------------------------
st.set_page_config(page_title="ResumeAI Pro", layout="centered")

st.title("📄 ResumeAI Pro - AI Resume Analyzer")

st.write("Upload your **Resume** and **Job Description** PDFs to analyze compatibility.")

uploaded_resume = st.file_uploader("Upload Resume (PDF)", type=["pdf"])
job_desc = st.file_uploader("Upload Job Description (PDF)", type=["pdf"])

# -----------------------------
# MAIN ANALYSIS
# -----------------------------
if st.button("Analyze Resume"):

    if uploaded_resume is None or job_desc is None:
        st.warning("⚠️ Please upload both Resume and Job Description")
    
    else:
        resume_text = extract_text_from_pdf(uploaded_resume)
        jd_text = extract_text_from_pdf(job_desc)

        if resume_text.strip() == "" or jd_text.strip() == "":
            st.error("❌ Could not extract text from one of the PDFs")
        
        else:
            # Preprocess
            resume_clean = preprocess(resume_text)
            jd_clean = preprocess(jd_text)

            # -----------------------------
            # SIMILARITY SCORE
            # -----------------------------
            vectorizer = TfidfVectorizer(ngram_range=(1,2))
            vectors = vectorizer.fit_transform([resume_clean, jd_clean])

            similarity = cosine_similarity(vectors[0], vectors[1])[0][0]

            # -----------------------------
            # KEYWORD MATCHING
            # -----------------------------
            jd_keywords = extract_keywords(jd_clean)
            jd_keywords = jd_keywords - BAD_WORDS

            resume_words = set(resume_clean.split())

            missing = []
            for word in jd_keywords:
                if word not in resume_words and len(word) > 3:
                    missing.append(word)

            # -----------------------------
            # FINAL SCORE (COMBINED)
            # -----------------------------
            matched = len(jd_keywords) - len(missing)
            skill_score = (matched / len(jd_keywords)) * 100 if len(jd_keywords) > 0 else 0

            final_score = (similarity * 100 * 0.4) + (skill_score * 0.6)

            if len(missing) == 0:
                final_score = min(final_score + 10, 95)

            score = round(final_score, 2)

            # -----------------------------
            # DISPLAY RESULTS
            # -----------------------------
            st.subheader(f"📊 Match Score: {score}%")

            if len(missing) == 0:
                st.success("✅ No major skills missing. Strong match!")
            else:
                st.subheader("❌ Missing Key Skills:")
                st.write(", ".join(missing))

            # Feedback
            if score < 50:
                st.error("🚨 Low match. Consider adding relevant skills.")
            elif score < 75:
                st.warning("⚠️ Moderate match. Improve alignment with job description.")
            else:
                st.success("🎯 Good match! Your resume aligns well.")

            # Suggestions
            if score < 85:
                st.subheader("💡 Suggestions:")
                st.write("- Use keywords exactly as in job description")
                st.write("- Add measurable achievements")
                st.write("- Include more relevant tools/technologies")

            # Preview
            st.subheader("📄 Resume Preview:")
            st.write(resume_text[:500])

            st.subheader("📄 Job Description Preview:")
            st.write(jd_text[:500])
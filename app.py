import streamlit as st
import PyPDF2
import nltk
import string

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.corpus import stopwords

# -----------------------------
# NLTK SETUP
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
            content = page.extract_text()
            if content:
                text += content
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
    return " ".join([w for w in words if w not in stop_words])

# -----------------------------
# FILTERS
# -----------------------------
SECTION_WORDS = {
    "core","skills","technologies","domains","tasks",
    "responsibilities","requirements","title","job",
    "summary","role"
}

NON_SKILL_CONCEPTS = {
    "system","application","process","task","role","team",
    "project","work","company","business","management",
    "skills","knowledge","experience","ability",
    "database","integration","optimization","performance",
    "design","development", "intern", "tools"
}

KNOWN_SKILLS = {
    # Tech
    "python","java","sql","html","css","javascript",
    "algorithms","structures","oop","git","github",
    "flask","streamlit","numpy","pandas","api","rest",

    # Business / other domains
    "marketing","seo","branding","finance","accounting",
    "cad","matlab","thermodynamics","excel","powerbi"
}

# -----------------------------
# SKILL VALIDATION
# -----------------------------
def is_valid_skill(word):
    if len(word) <= 3:
        return False
    if word.endswith(("ing","ed","ly")):
        return False
    return True

# -----------------------------
# KEYWORD EXTRACTION
# -----------------------------
def extract_keywords(text):
    vectorizer = TfidfVectorizer(stop_words='english', max_features=100)
    vectorizer.fit([text])
    words = vectorizer.get_feature_names_out()

    filtered = {
        w for w in words
        if (
            (is_valid_skill(w) and w not in SECTION_WORDS and w not in NON_SKILL_CONCEPTS)
            or w in KNOWN_SKILLS
        )
    }

    return filtered

# -----------------------------
# STREAMLIT UI
# -----------------------------
st.set_page_config(page_title="ResumeAI Pro", layout="centered")

st.title("📄 ResumeAI Pro - Smart Resume Analyzer")

st.write("Upload Resume and Job Description PDFs")

resume_file = st.file_uploader("Upload Resume (PDF)", type=["pdf"])
jd_file = st.file_uploader("Upload Job Description (PDF)", type=["pdf"])

# -----------------------------
# MAIN LOGIC
# -----------------------------
if st.button("Analyze"):

    if resume_file is None or jd_file is None:
        st.warning("Please upload both files")

    else:
        resume_text = extract_text_from_pdf(resume_file)
        jd_text = extract_text_from_pdf(jd_file)

        if not resume_text or not jd_text:
            st.error("Text extraction failed. Try another PDF.")

        else:
            # Clean text
            resume_clean = preprocess(resume_text)
            jd_clean = preprocess(jd_text)

            # -----------------------------
            # SIMILARITY
            # -----------------------------
            vectorizer = TfidfVectorizer(ngram_range=(1,2))
            vectors = vectorizer.fit_transform([resume_clean, jd_clean])

            similarity = cosine_similarity(vectors[0], vectors[1])[0][0]

            # -----------------------------
            # SKILL EXTRACTION
            # -----------------------------
            jd_skills = extract_keywords(jd_clean)
            resume_words = set(resume_clean.split())

            # -----------------------------
            # MISSING SKILLS
            # -----------------------------
            missing = [
                skill for skill in jd_skills
                if skill not in resume_words
            ]

            # -----------------------------
            # SCORING
            # -----------------------------
            if len(jd_skills) > 0:
                skill_score = ((len(jd_skills) - len(missing)) / len(jd_skills)) * 100
            else:
                skill_score = 0

            final_score = (similarity * 100 * 0.4) + (skill_score * 0.6)

            if len(missing) == 0:
                final_score = min(final_score + 10, 95)

            score = round(final_score, 2)

            # -----------------------------
            # OUTPUT
            # -----------------------------
            st.subheader(f"📊 Match Score: {score}%")

            if missing:
                st.subheader("❌ Missing Skills:")
                st.write(", ".join(sorted(missing)))
            else:
                st.success("✅ No major skills missing")

            # Feedback
            if score < 50:
                st.error("Low match. Add relevant skills.")
            elif score < 75:
                st.warning("Moderate match. Improve alignment.")
            else:
                st.success("Good match!")

            # Suggestions
            st.subheader("💡 Suggestions:")
            st.write("- Include more relevant tools and technologies")
            st.write("- Match keywords from job description")
            st.write("- Add project experience using required skills")

            # Preview
            st.subheader("📄 Resume Preview")
            st.write(resume_text[:399])

            st.subheader("📄 Job Description Preview")
            st.write(jd_text[:399])

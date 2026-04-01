import PyPDF2
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.corpus import stopwords
import string

try:
    stopwords.words('english')
except LookupError:
    nltk.download('stopwords')

def extract_text_from_pdf(uploaded_file):
    text = ""
    pdf_reader = PyPDF2.PdfReader(uploaded_file)
    
    for page in pdf_reader.pages:
        extracted = page.extract_text()
        if extracted:
            text += extracted
    
    return text

def extract_top_keywords(text, n=20):
    vectorizer = TfidfVectorizer(
        stop_words='english',
        ngram_range=(1,2),
        max_features=50
    )
    
    vectors = vectorizer.fit_transform([text])
    return set(vectorizer.get_feature_names_out())

def preprocess(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))  # remove punctuation
    words = text.split()
    stop_words = set(stopwords.words('english'))
    words = [w for w in words if w not in stop_words]
    return " ".join(words)

def extract_top_keywords(text, n=20):
    vectorizer = TfidfVectorizer(
        stop_words='english',
        max_features=100
    )
    
    vectorizer.fit([text])
    keywords = vectorizer.get_feature_names_out()
    
    return set(keywords)

st.title("ResumeAI Pro - AI Resume Planner")

uploaded_resume = st.file_uploader("Upload Resume (PDF)", type=["pdf"])
job_desc = st.file_uploader("Upload Job Description (PDF)", type=["pdf"])

if st.button("Analyze Resume"):
    if uploaded_resume is None or job_desc is None:
        st.warning("Enter both fields")
    else:
        resume_text = extract_text_from_pdf(uploaded_resume)
        resume_clean = preprocess(resume_text)
        jd_text = extract_text_from_pdf(job_desc)
        jd_clean = preprocess(jd_text)

        vectorizer = TfidfVectorizer(ngram_range=(1,2))
        vectors = vectorizer.fit_transform([resume_clean, jd_clean])

        similarity = cosine_similarity(vectors[0], vectors[1])[0][0]
        
 COMMON_WORDS = {
    # generic job words
    "job","role","candidate","position","company","organization",

    # vague qualities
    "strong","good","excellent","basic","advanced","solid","proven",

    # soft skills (remove to avoid noise)
    "communication","team","teamwork","leadership","collaboration",
    "analytical","problem","solving","skills","ability",

    # filler verbs
    "work","working","worked","develop","developed","developing",
    "maintain","maintaining","build","building","create","creating",
    "write","writing","optimize","optimizing","debug","debugging", "implement",

    # job description fillers
    "responsibilities","requirements","required","preferred",
    "including","such","various","related","field",

    # education noise
    "degree","bachelor","master","science","engineering","computer",

    # connectors / useless
    "using","based","with","and","or","for","the","a","an","to","of"
}

        jd_keywords = extract_top_keywords(jd_clean, n=30)
        jd_keywords = {
        word for word in jd_keywords
        if word not in COMMON_WORDS and len(word) > 3
        }
        resume_keywords = set(resume_clean.split())

        missing = []

        for word in jd_keywords:
            if word not in resume_keywords:
        # Ignore very short or useless words
                if len(word) > 3:
                     missing.append(word)
        
        matched = len(jd_keywords) - len(missing)
        skill_score = (matched / len(jd_keywords)) * 100

        final_score = (similarity * 100 * 0.4) + (skill_score * 0.6)
        score = round(final_score, 2)

        
        if len(missing) == 0:
         score = min(score + 10, 95)

        st.subheader(f"Match Score: {score}%")

        st.subheader("Missing Key Skills:")
        
        if len(missing) == 0:
            st.success("No major skills missing. Resume matches job requirements well.")
            if score < 85:
                 st.subheader("Suggestions:")
                 st.write("Try aligning your resume wording more closely with the job description.")
                 st.write("Include exact terms like 'develop', 'optimize', 'collaborate'.")
        
             
        else:
            st.write(", ".join(missing))

        st.subheader("Extracted Resume Text (Preview):")
        st.write(resume_text[:500])


        if score < 50:
            st.error("Low match. Improve skills.")
        elif score < 75:
            st.warning("Moderate match.")
        else:
            st.success("Good match!")

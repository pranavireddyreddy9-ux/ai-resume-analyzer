import streamlit as st
import PyPDF2
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def extract_text_from_pdf(pdf_file):

    text = ""

    pdf_reader = PyPDF2.PdfReader(pdf_file)

    for page in pdf_reader.pages:

        extracted = page.extract_text()

        if extracted:
            text += extracted

    return text


st.title("AI Resume Analyzer")

uploaded_resume = st.file_uploader(
    "Upload Resume PDF",
    type=["pdf"]
)

job_description = st.text_area(
    "Paste Job Description"
)

if uploaded_resume and job_description:

    resume_text = extract_text_from_pdf(uploaded_resume)

    documents = [resume_text, job_description]

    vectorizer = CountVectorizer()

    matrix = vectorizer.fit_transform(documents)

    similarity = cosine_similarity(matrix)[0][1]

    match_percentage = round(similarity * 100, 2)

    st.success(
        f"Resume Match Score: {match_percentage}/100"
    )

    common_skills = [
        "python",
        "sql",
        "machine learning",
        "data analysis",
        "power bi"
    ]

    missing_skills = []

    resume_lower = resume_text.lower()

    jd_lower = job_description.lower()

    for skill in common_skills:

        if skill in jd_lower and skill not in resume_lower:

            missing_skills.append(skill)

    st.subheader("Missing Skills")

    if missing_skills:

        for skill in missing_skills:

            st.write(f"- {skill}")

    else:

        st.write("No major missing skills detected.")
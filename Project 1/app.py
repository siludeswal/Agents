import streamlit as st
from pypdf import PdfReader

from resumeshortlister import graph

st.title(" Resume Shortlister Model")

file = st.file_uploader(
    "Upload Resume PDF",
    type=["pdf"]
)

if file:

    st.success("Resume Uploaded Successfully")

    reader = PdfReader(file)

    content = ""

    for page in reader.pages:
        text = page.extract_text()

        if text:
            content += text + "\n"

    if st.button("Analyze Resume"):

        with st.spinner("Analyzing Resume..."):

            response = graph.invoke(
                {
                    "resume_text": content
                }
            )

        st.subheader("Resume Information")
        st.write(response.get("resume_information", ""))

        st.subheader("Personal Information")
        st.write(response.get("personal_information", ""))

        st.subheader("Skills")
        st.write(response.get("skill", ""))

        st.subheader("Score")
        st.metric(
            "Resume Score",
            response.get("score", 0)
        )

        decision = response.get(
            "final_decision",
            "No Decision"
        )

        st.subheader("Final Decision")

        if "select" in str(decision).lower():
            st.success(decision)
        else:
            st.error(decision)
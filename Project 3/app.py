import streamlit as st
from model import graph
st.title("Medical AI Assistance")

query = st.text_area("Enter the symptoms and Your Address",height=50)
if st.button("Generate Report"):
    if query:
        with st.spinner("Analyzing your input..."):
            result = graph.invoke({"collect_input":query})

        st.subheader("Your Medical Report:")
        st.write(result.get("final_report"))
    else:
        st.warning("please write your query")



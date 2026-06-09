import streamlit as st
from CodeDebugger import graph

st.title("Code Debugger Model")

code = st.text_area("Paste your code here", height=300)

if st.button("Debug"):
    if code.strip():
        with st.spinner("Analyzing your code..."):
            response = graph.invoke({
                "input_code": code
            })

        st.subheader("Updated Version of your code:")
        st.write(response.get("final_observations", ""))
    else:
        st.warning("Please enter some code.")
import streamlit as st

st.set_page_config(page_title="AI Services", layout="wide")

st.markdown("<h1 style='text-align: center; color: #00FFAA;'>AI Services</h1>", unsafe_allow_html=True)
st.markdown("### Explore our AI-powered tools below:")

col1, col2 = st.columns(2)

with col1:
    st.markdown("#### 📊 CSV Query")
    st.write("Upload an Excel/CSV file and ask questions from your data.")
    st.page_link("pages/1_CSV_Query.py", label="Go to CSV Query", icon="➡️")

    st.markdown("#### 📄 Table Extraction")
    st.write("Extract structured tables from scanned or digital PDFs.")
    st.page_link("pages/2_Table_Extraction.py", label="Go to Table Extraction", icon="➡️")

with col2:
    st.markdown("#### 🔍 Data Point Extraction")
    st.write("Ask specific data points from your uploaded documents.")
    st.page_link("pages/3_Data_Point_Extraction.py", label="Go to Data Point Extraction", icon="➡️")

    st.markdown("#### 🧠 NLQ to SQL")
    st.write("Ask questions and convert them into SQL queries.")
    st.page_link("pages/4_NLQ_to_SQL.py", label="Go to NLQ to SQL", icon="➡️")







import streamlit as st

st.set_page_config(page_title="NLQ to SQL", layout="wide")

st.title("🧠 NLQ to SQL")
st.markdown("Ask a natural language question and convert it into SQL.")

query = st.text_input("Enter your question (e.g., Show all employees older than 30):")

if st.button("Convert to SQL"):
    st.info("Your SQL output will appear here (pending backend integration).")

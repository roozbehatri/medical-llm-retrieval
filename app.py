import streamlit as st
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_openai import OpenAI, OpenAIEmbeddings
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

st.set_page_config(page_title="Medical Report QA & Summarization", layout="wide")

st.title("🩺 Medical Report Q&A & Summarization App")

@st.cache_resource
def load_vectorstore():
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.load_local(
        "results/faiss_vectorstore",
        embeddings,
        allow_dangerous_deserialization=True
    )
    return vectorstore

vectorstore = load_vectorstore()
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
llm = OpenAI(temperature=0)

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    return_source_documents=True
)

# ✅ Create Tabs
tab1, tab2 = st.tabs(["🧐 Question Answering", "📝 Summarization"])

# -------------------------------------------------------
# ✅ QA TAB
with tab1:
    st.subheader("Ask a Question about the Medical Reports")

    # Pre-suggested questions
    st.markdown("#### Suggested Questions:")
    col1, col2 = st.columns(2)

    with col1:
        if st.button("What are the key findings?"):
            user_query = "What are the key findings in this chest X-ray report?"
        elif st.button("Is the chest X-ray normal?"):
            user_query = "Is the chest X-ray normal?"
        else:
            user_query = st.text_input("Or type your own question:")

    with col2:
        if st.button("Is there evidence of pneumonia?"):
            user_query = "Is there evidence of pneumonia in this report?"
        elif st.button("Are there signs of heart enlargement?"):
            user_query = "Are there signs of heart enlargement?"

    if st.button("Get Answer") and user_query:
        with st.spinner("Retrieving answer..."):
            result = qa_chain(user_query)

        st.subheader("Answer:")
        st.write(result['result'])

        st.subheader("Source Documents:")
        for doc in result['source_documents']:
            st.write(f"Source: {doc.metadata}")
            st.write(f"Excerpt: {doc.page_content[:300]}...")
            st.write("---")

# -------------------------------------------------------
# ✅ Summarization TAB
with tab2:
    st.subheader("Paste a Chest X-Ray Report to Summarize")

    report_text = st.text_area("Paste the findings section here:")

    if st.button("Summarize Report"):
        if report_text.strip() != "":
            with st.spinner("Summarizing..."):
                from langchain.prompts import PromptTemplate
                summarization_prompt = PromptTemplate(
                    input_variables=["report"],
                    template="Please provide a concise summary of the following chest X-ray findings:\n\n{report}\n\nSummary:"
                )

                summarization_chain = summarization_prompt | llm
                summary = summarization_chain.invoke({"report": report_text})

            st.subheader("Summary:")
            st.write(summary)
        else:
            st.warning("Please paste a report to summarize.")
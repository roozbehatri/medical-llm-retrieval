# 🩺 Medical Report QA and Summarization App

This project is an interactive **Streamlit web application** that enables question-answering (QA) and summarization for chest X-ray medical reports using **LangChain**, **OpenAI's GPT models**, and **vector search with FAISS**.

It demonstrates how to combine retrieval-augmented generation (RAG) with summarization techniques to explore and understand medical text data efficiently.

---

## 🚀 Features

- 🔍 **Question Answering:**  
  Ask natural language questions about chest X-ray reports using a retrieval-based QA system powered by FAISS and OpenAI GPT.

- ✍️ **Summarization:**  
  Automatically summarize the findings sections of chest X-ray reports using OpenAI’s GPT.

- 💡 **Pre-Suggested Questions:**  
  Convenient one-click suggested questions to test the retrieval system quickly.

- 🖥️ **Streamlit Web App:**  
  Fully interactive, browser-based app with multi-tab interface.

---

## 🛠️ Technologies Used

- [Streamlit](https://streamlit.io/) for building the web app
- [LangChain](https://python.langchain.com/) for chaining LLMs and retrievers
- [FAISS](https://github.com/facebookresearch/faiss) for fast similarity search
- [OpenAI GPT](https://platform.openai.com/) for answering and summarizing
- [Python](https://www.python.org/)
- [Pandas](https://pandas.pydata.org/) for data handling

---

## 📂 Project Structure

```text
medical-report-qa-langchain/
├── app.py                # Streamlit app
├── data/                 # Processed and cleaned datasets
├── results/              # FAISS vector store and outputs
├── notebooks/            # Exploratory and preprocessing notebooks
├── requirements.txt      # Python package dependencies
├── README.md             # Project documentation
└── .env                  # API keys (not tracked by Git)
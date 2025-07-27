

# 🚀 Advanced RAG Q\&A System with Real-Time Evaluation

An intelligent, interactive document Q\&A application built with **Python** and **Streamlit**. This system leverages a sophisticated **Retrieval-Augmented Generation (RAG)** pipeline to provide accurate answers from your uploaded **PDF documents** and includes a built-in **evaluation module** to measure retrieval performance in real-time.

---

## ✨ Live Demo



![WhatsAppVideo2025-07-27at7 03 07PM-ezgif com-resize (1)](https://github.com/user-attachments/assets/e38fc411-2108-461f-ab7e-e732ccd0c707)



---

## 🔑 Key Features

* **📁 Dynamic Document Upload:** Upload multiple PDFs directly through the web interface.
* **🤖 Intelligent Q\&A:** Ask natural language questions and receive concise, context-aware answers.
* **🧠 Retrieval-Augmented Generation (RAG):** Uses a cutting-edge RAG pipeline for grounded and reliable responses.
* **📊 Advanced Evaluation Suite:**

  * Add and manage a suite of test cases.
  * Evaluate the retrieval system on-the-fly using **Hit Rate** and **Mean Reciprocal Rank (MRR)**.
* **🖥️ Interactive UI:** Clean, user-friendly interface powered by Streamlit.

---

## ⚙️ How It Works

This project implements a **complete RAG pipeline** that transforms your documents into a queryable knowledge base.

### 1. **Document Loading & Preprocessing**

* PDF files are uploaded.
* Text is extracted and cleaned by removing special characters and extra whitespaces.

### 2. **Advanced Text Chunking**

* **Parent Chunks:** Large, overlapping chunks providing broad context.
* **Child Chunks:** Smaller segments optimized for semantic search.

### 3. **Embedding & Indexing**

* Chunks are embedded using `all-MiniLM-L6-v2`.
* Stored in **ChromaDB** for fast similarity search.

### 4. **Retrieval**

* The query is embedded and compared against stored embeddings.
* **Top K** most relevant chunks are retrieved.

### 5. **Context Augmentation**

* Corresponding parent chunks are fetched to enrich context.

### 6. **Answer Generation**

* The user's query and the retrieved context are sent to **llama3-8b-8192** (via **Groq API**).
* A context-aware answer is generated and displayed.

---

## 📈 Advanced Evaluation Metrics Explained

### ✅ **Hit Rate**

**What is it?**
Measures how often the correct document is in the top K retrieved results.

**Why it matters:**
If the right document isn't retrieved, even the best language model can't give the right answer.

**Interpretation Example:**
A **Hit Rate of 95%** means 95 out of 100 test cases retrieved the correct document.

---

### 📉 **Mean Reciprocal Rank (MRR)**

**What is it?**
Measures the average rank position of the correct document (higher is better).

**Why it matters:**
Ranks matter — documents near the top have higher influence on the generated answer.

**Interpretation Examples:**

* `MRR = 1.0` → Correct doc always ranked 1st (Perfect).
* `MRR = 0.75` → Correct doc usually near the top.
* `MRR = 0.25` → Correct doc found, but ranked low.

---

## 🛠️ Tech Stack

| Component          | Technology                                 |
| ------------------ | ------------------------------------------ |
| **Language**       | Python                                     |
| **Frontend**       | Streamlit                                  |
| **LLM API**        | Groq (`llama3-8b-8192`)                    |
| **Vector Store**   | ChromaDB (in-memory)                       |
| **Embeddings**     | `all-MiniLM-L6-v2` (sentence-transformers) |
| **PDF Processing** | pypdf                                      |
| **Text Splitting** | LangChain                                  |

---

## 🚀 Setup and Installation

### 1. **Clone the Repository**

```bash
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name
```

### 2. **Create a Virtual Environment**

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. **Install Dependencies**

```bash
pip install -r requirements.txt
```

### 4. **Add Your API Key**

Create a `.env` file in the root directory:

```env
GROQ_API_KEY="your-groq-api-key-here"
```

---

## 🏃‍♀️ Usage Instructions

### ✅ Run the App

```bash
streamlit run app.py
```

### 📁 Upload Documents

* Use the **sidebar file uploader** to select one or more PDFs.
* Click **"Process Uploaded Documents"** and wait for processing.

### ❓ Ask a Question

* Navigate to the **"Ask a Question"** page.
* Enter your question and click **"Get Answer"**.

### 📊 Add Test Cases & Evaluate

* Go to **"Add a Test Case"** to enter question-answer pairs.
* Use **"Evaluation"** page and click **"Run Evaluation"** to compute **Hit Rate** and **MRR**.

---


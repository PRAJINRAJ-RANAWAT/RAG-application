🚀 Advanced RAG Q&A System with Real-Time Evaluation
An intelligent, interactive document Q&A application built with Python and Streamlit. This system leverages a sophisticated Retrieval-Augmented Generation (RAG) pipeline to provide accurate answers from your uploaded PDF documents and includes a built-in evaluation module to measure the performance of the retrieval system in real-time.

✨ Live Demo
A quick demonstration of the application in action.

(Note: Replace this with a GIF of your running application for maximum impact!)

🔑 Key Features
Dynamic Document Upload: Upload multiple PDF files directly through the web interface.

Intelligent Q&A: Ask questions in natural language and receive concise, context-aware answers.

Retrieval-Augmented Generation (RAG): Utilizes a state-of-the-art RAG pipeline to ensure answers are grounded in the provided documents.

Advanced Evaluation Suite:

Add and manage a suite of test cases.

Evaluate the retrieval system on the fly using Hit Rate and Mean Reciprocal Rank (MRR).

Interactive UI: A clean, user-friendly interface powered by Streamlit.

⚙️ How It Works
This project implements a complete RAG pipeline to transform your documents into a queryable knowledge base.

Document Loading & Pre-processing: PDF files are uploaded and their text content is extracted. The text is cleaned by removing special characters and extra whitespace.

Advanced Text Chunking: The documents are split into two levels of chunks:

Parent Chunks: Larger, overlapping chunks that provide broad context for the language model.

Child Chunks: Smaller, more granular chunks derived from the parent chunks, optimized for semantic search.

Embedding & Indexing: The child chunks are converted into vector embeddings using the all-MiniLM-L6-v2 model and indexed in a ChromaDB vector store for efficient similarity searches.

Retrieval: When a question is asked, it's embedded and used to query the ChromaDB. The system retrieves the TOP_K most relevant child chunks.

Context Augmentation: The parent chunks corresponding to the retrieved child chunks are fetched. This provides richer, more complete context to the language model.

Answer Generation: The user's question and the augmented context are passed to a powerful Large Language Model (in this case, llama3-8b-8192 via the Groq API). The model generates a precise answer based only on the provided context.

📊 Advanced Evaluation Metrics Explained
A standout feature of this application is its ability to rigorously evaluate the performance of the retrieval system. This is crucial for understanding how well the system finds the correct information before it even reaches the language model.

Hit Rate
What is it?
The Hit Rate measures how often the retrieval system successfully finds the correct source document within the top K results for a given question. In simpler terms, it answers the question: "Did we find the right document at all?"

Why is it Important?
A high Hit Rate indicates that your retrieval system is effective at identifying the relevant documents for a query. If the correct document isn't retrieved, the language model won't have the necessary context to generate a correct answer, no matter how powerful it is.

How to Interpret It:
A Hit Rate of 95% means that for 95 out of 100 test questions, the correct source document was among the top results retrieved by the system.

Mean Reciprocal Rank (MRR)
What is it?
Mean Reciprocal Rank (MRR) is a more nuanced metric that evaluates not just if the correct document was found, but also how high up in the search results it was. It is the average of the reciprocal ranks of the correct answers. The reciprocal rank is 1 / rank.

Why is it Important?
MRR rewards systems that place the correct document closer to the top of the results list. This is critical because higher-ranked documents are more likely to provide the most relevant context to the language model. A system can have a 100% Hit Rate but a low MRR if it consistently finds the correct document but ranks it last.

How to Interpret It:

An MRR of 1.0 is a perfect score, meaning the correct document was always the #1 result.

An MRR of 0.75 indicates that, on average, the correct document is ranked fairly high.

An MRR of 0.25 suggests that the correct documents are often found but are buried deep in the search results.

🛠️ Tech Stack
Language: Python

Web Framework: Streamlit

LLM API: Groq (for llama3-8b-8192)

Vector Database: ChromaDB (in-memory)

Embedding Model: sentence-transformers (all-MiniLM-L6-v2)

PDF Processing: pypdf

Text Splitting: langchain

🚀 Setup and Installation
Follow these steps to get the application running on your local machine.

Clone the Repository:

git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name

Create a Virtual Environment:

python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`

Install Dependencies:

pip install -r requirements.txt

Set Up Environment Variables:
Create a file named .env in the root of your project directory and add your Groq API key:

GROQ_API_KEY="your-groq-api-key-here"

🏃‍♀️ Usage
Run the Streamlit App:

streamlit run app.py

Upload Documents:

In the sidebar, use the file uploader to select one or more PDF documents.

Click the "Process Uploaded Documents" button and wait for the processing to complete.

Ask a Question:

Navigate to the "Ask a Question" page.

Type your question into the text box and click "Get Answer".

Add Test Cases & Evaluate:

Go to the "Add a Test Case" page to build your evaluation suite.

Navigate to the "Evaluation" page and click "Run Evaluation" to see the Hit Rate and MRR for your retrieval system.

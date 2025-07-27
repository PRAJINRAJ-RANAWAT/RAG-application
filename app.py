import os
import re
import streamlit as st
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import chromadb
from groq import Groq
from pypdf import PdfReader
from tqdm import tqdm
import pandas as pd
import shutil
import tempfile

st.set_page_config(page_title="Document Q&A with RAG", layout="wide")

load_dotenv()

MODEL_NAME = "all-MiniLM-L6-v2"
CHILD_CHUNK_SIZE = 128
PARENT_CHUNK_SIZE = 512
GROQ_MODEL = "llama3-8b-8192"
TOP_K = 3

@st.cache_resource
def initialize_embedding_and_groq():
    try:
        embedding_model = SentenceTransformer(MODEL_NAME)
        groq_api_key = os.environ.get("GROQ_API_KEY")
        if not groq_api_key:
            st.error("GROQ_API_KEY not found. Please set it in your .env file.")
            return None, None
        groq_client = Groq(api_key=groq_api_key)
        return embedding_model, groq_client
    except Exception as e:
        st.error(f"Error initializing models: {e}")
        return None, None

embedding_model, groq_client = initialize_embedding_and_groq()

@st.cache_resource
def initialize_chroma():
    try:
        temp_dir = tempfile.mkdtemp()
        chroma_client = chromadb.PersistentClient(path=temp_dir)
        try:
            collection = chroma_client.get_collection(name="document_chunks")
            st.info("Found existing collection. Clearing it for new data.")
            chroma_client.delete_collection(name="document_chunks")
        except:
            pass
        collection = chroma_client.get_or_create_collection(
            name="document_chunks",
            metadata={"hnsw:space": "cosine"}
        )
        return chroma_client, collection
    except Exception as e:
        st.error(f"Error initializing ChromaDB: {e}")
        st.info("Falling back to in-memory ChromaDB...")
        chroma_client = chromadb.Client()
        collection = chroma_client.get_or_create_collection(
            name="document_chunks",
            metadata={"hnsw:space": "cosine"}
        )
        return chroma_client, collection

if 'chroma_initialized' not in st.session_state:
    chroma_client, collection = initialize_chroma()
    st.session_state.chroma_client = chroma_client
    st.session_state.collection = collection
    st.session_state.chroma_initialized = True
else:
    chroma_client = st.session_state.chroma_client
    collection = st.session_state.collection

def reset_chroma():
    try:
        try:
            st.session_state.chroma_client.delete_collection(name="document_chunks")
        except:
            pass
        collection = st.session_state.chroma_client.get_or_create_collection(
            name="document_chunks",
            metadata={"hnsw:space": "cosine"}
        )
        st.session_state.collection = collection
        return collection
    except Exception as e:
        st.error(f"Error resetting ChromaDB: {e}")
        return None

def process_documents(uploaded_files):
    collection = reset_chroma()
    if collection is None:
        st.error("Failed to reset database collection.")
        return [], {}

    parent_chunks = []
    child_to_parent = {}

    text_splitter_parent = RecursiveCharacterTextSplitter(
        chunk_size=PARENT_CHUNK_SIZE, chunk_overlap=20, separators=["\n\n", "\n", ". "]
    )
    text_splitter_child = RecursiveCharacterTextSplitter(
        chunk_size=CHILD_CHUNK_SIZE, chunk_overlap=0, separators=["\n", ". ", "! ", "? "]
    )

    child_counter = 0
    progress_bar = st.progress(0, text="Processing documents...")
    
    for i, uploaded_file in enumerate(uploaded_files):
        filename = uploaded_file.name
        try:
            reader = PdfReader(uploaded_file)
            text = "\n".join([page.extract_text() for page in reader.pages if page.extract_text()])
            text = re.sub(r'\s+', ' ', text)
            text = re.sub(r'[^\w\s.,;:!?]', '', text)

            parents = text_splitter_parent.split_text(text)
            for parent_idx, parent in enumerate(parents):
                parent_id = f"{filename}_parent_{parent_idx}"
                parent_chunks.append({
                    "text": parent,
                    "source": filename,
                    "parent_id": parent_id
                })

                children = text_splitter_child.split_text(parent)
                child_documents = []
                child_ids = []
                child_metadatas = []
                
                for child in children:
                    if child.strip():
                        child_id = f"child_{child_counter}"
                        child_documents.append(child)
                        child_ids.append(child_id)
                        child_metadatas.append({"parent_id": parent_id})
                        child_to_parent[child_id] = len(parent_chunks) - 1
                        child_counter += 1
                
                if child_documents:
                    try:
                        collection.add(
                            documents=child_documents,
                            ids=child_ids,
                            metadatas=child_metadatas
                        )
                    except Exception as e:
                        st.error(f"Error adding documents to collection: {e}")
                        continue

        except Exception as e:
            st.error(f"Error processing {filename}: {e}")
        
        progress_bar.progress((i + 1) / len(uploaded_files), text=f"Processing {filename}...")
    
    progress_bar.empty()
    
    if parent_chunks:
        st.success(f"Successfully processed {len(parent_chunks)} parent chunks from {len(uploaded_files)} documents!")
    else:
        st.warning("No documents were processed successfully.")
    
    return parent_chunks, child_to_parent

if 'parent_chunks' not in st.session_state:
    st.session_state.parent_chunks = []
if 'child_to_parent' not in st.session_state:
    st.session_state.child_to_parent = {}

def retrieve_parent_chunks(question):
    parent_chunks = st.session_state.parent_chunks
    child_to_parent = st.session_state.child_to_parent
    collection = st.session_state.collection
    
    if not parent_chunks:
        return []
    
    try:
        count = collection.count()
        if count == 0:
            st.warning("No documents in the database. Please process documents first.")
            return []
        
        question = re.sub(r'[^\w\s.,;:!?]', '', question)
        results = collection.query(query_texts=[question], n_results=min(TOP_K, count))
        
        parent_indices = set()
        if results and results['ids'] and results['ids'][0]:
            for child_id in results['ids'][0]:
                parent_idx = child_to_parent.get(child_id)
                if parent_idx is not None and parent_idx < len(parent_chunks):
                    parent_indices.add(parent_idx)
        
        return [parent_chunks[i] for i in parent_indices]
    except Exception as e:
        st.error(f"Error retrieving chunks from the database: {e}")
        return []

def generate_answer(question, context_chunks):
    if not context_chunks:
        return "I couldn't find any relevant information in the documents."
        
    context = "\n\n".join([f"Source: {chunk['source']}\n{chunk['text']}" for chunk in context_chunks])
    system_prompt = f"""
    You are an expert document assistant. Answer the user's question using ONLY the provided context.
    If the answer isn't in the context, say "I don't know". Be concise and accurate.

    Context:
    {context}
    """
    try:
        response = groq_client.chat.completions.create(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": question}
            ],
            model=GROQ_MODEL,
            temperature=0.1,
            max_tokens=512
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error generating answer: {e}"

def ask_question(question):
    context_chunks = retrieve_parent_chunks(question)
    return generate_answer(question, context_chunks), context_chunks

class EvaluationSystem:
    def __init__(self, test_questions):
        self.test_questions = test_questions
    
    def evaluate(self):
        hit_count = 0
        reciprocal_ranks = []
        
        progress_bar = st.progress(0, text="Evaluating test cases...")
        for i, test_case in enumerate(tqdm(self.test_questions, desc="Evaluating", leave=False)):
            context_chunks = retrieve_parent_chunks(test_case["question"])
            sources = [chunk["source"] for chunk in context_chunks]
            if test_case["expected_source"] in sources:
                hit_count += 1
                rank = sources.index(test_case["expected_source"]) + 1
                reciprocal_ranks.append(1/rank)
            else:
                reciprocal_ranks.append(0)
            progress_bar.progress((i + 1) / len(self.test_questions), text=f"Evaluating case {i+1}...")

        hit_rate = hit_count / len(self.test_questions) if self.test_questions else 0
        mrr = sum(reciprocal_ranks) / len(self.test_questions) if self.test_questions else 0
        
        progress_bar.empty()
        return hit_rate, mrr

st.title("📄 Document Q&A with RAG")
st.markdown("Ask questions about your documents, and the system will find the answers.")

if 'test_cases' not in st.session_state:
    st.session_state.test_cases = []
if 'processed' not in st.session_state:
    st.session_state.processed = False

with st.sidebar:
    st.header("Navigation")
    page = st.radio("Go to", ["Ask a Question", "Add a Test Case", "Evaluation"])

    st.markdown("---")
    st.header("Setup")
    st.info("Please ensure you have a `.env` file with your `GROQ_API_KEY`.")
    
    uploaded_files = st.file_uploader(
        "Upload your PDF documents", 
        type="pdf", 
        accept_multiple_files=True
    )

    if uploaded_files:
        if st.button("Process Uploaded Documents"):
            with st.spinner("Processing documents..."):
                parent_chunks, child_to_parent = process_documents(uploaded_files)
                st.session_state.parent_chunks = parent_chunks
                st.session_state.child_to_parent = child_to_parent
                st.session_state.processed = len(parent_chunks) > 0
    
    if st.session_state.processed:
        st.success(f"✅ {len(st.session_state.parent_chunks)} chunks processed")
        try:
            doc_count = st.session_state.collection.count()
            st.info(f"📊 {doc_count} documents in database")
        except:
            st.warning("⚠️ Database connection issue")

if page == "Ask a Question":
    st.header("Ask a Question")
    user_question = st.text_input("Enter your question here:", "")
    if st.button("Get Answer"):
        if not user_question:
            st.warning("Please enter a question.")
        elif not st.session_state.get('processed'):
            st.error("No documents processed. Please upload and process documents in the sidebar.")
        else:
            with st.spinner("Finding an answer..."):
                answer, context_chunks = ask_question(user_question)
                st.subheader("Answer")
                st.markdown(answer)
                
                if context_chunks:
                    with st.expander("Show Retrieved Context"):
                        for i, chunk in enumerate(context_chunks, 1):
                            st.markdown(f"**Context {i} - Source:** {chunk['source']}")
                            st.markdown(f"> {chunk['text']}")
                            st.markdown("---")
                else:
                    st.info("No relevant context found in the documents.")

elif page == "Add a Test Case":
    st.header("Manage Test Cases")
    with st.form("test_case_form"):
        st.subheader("Add a New Test Case")
        question = st.text_input("Question")
        expected_source = st.text_input("Expected Source Document (e.g., my_doc.pdf)")
        expected_answer = st.text_area("Expected Answer")
        submitted = st.form_submit_button("Add Test Case")
        if submitted:
            if question and expected_source and expected_answer:
                st.session_state.test_cases.append({
                    "question": question,
                    "expected_source": expected_source,
                    "expected_answer": expected_answer
                })
                st.success("Test case added!")
            else:
                st.error("Please fill out all fields.")
    
    st.markdown("---")
    st.subheader("Current Test Cases")
    if not st.session_state.test_cases:
        st.info("No test cases added yet.")
    else:
        for i, tc in enumerate(st.session_state.test_cases):
            with st.expander(f"Test Case {i+1}: {tc['question']}"):
                st.markdown(f"**Expected Source:** `{tc['expected_source']}`")
                st.markdown(f"**Expected Answer:**\n> {tc['expected_answer']}")
                if st.button(f"Delete Test Case {i+1}", key=f"delete_{i}"):
                    st.session_state.test_cases.pop(i)
                    st.rerun()

elif page == "Evaluation":
    st.header("Evaluate Retrieval Performance")
    if not st.session_state.test_cases:
        st.warning("No test cases to evaluate. Please add some on the 'Add a Test Case' page.")
    elif not st.session_state.get('processed'):
        st.error("No documents processed. Please upload and process documents first.")
    else:
        st.info(f"Found {len(st.session_state.test_cases)} test cases.")
        if st.button("Run Evaluation"):
            eval_system = EvaluationSystem(st.session_state.test_cases)
            hit_rate, mrr = eval_system.evaluate()
            
            st.subheader("Evaluation Results")
            col1, col2 = st.columns(2)
            with col1:
                st.metric(label="Hit Rate", value=f"{hit_rate:.2%}")
                st.caption("Percentage of queries where the correct source document was in the top K results.")
            with col2:
                st.metric(label="Mean Reciprocal Rank (MRR)", value=f"{mrr:.4f}")
                st.caption("The average of the reciprocal ranks of the correct source document.")

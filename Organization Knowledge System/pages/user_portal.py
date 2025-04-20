# pages/user_portal.py
import streamlit as st
from PyPDF2 import PdfReader
import pandas as pd
import base64
from datetime import datetime
from pymongo import MongoClient
import json
from docx import Document
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# LangChain imports
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate

# Initialize MongoDB connection
def get_mongo_client():
    client = MongoClient("mongodb://localhost:27017/")
    return client

def initialize_db():
    client = get_mongo_client()
    db = client["knowledge_base"]
    return db

# Get departments
def get_departments(db):
    return list(db.departments.find())

# Load existing files from MongoDB based on department
def load_existing_files(db, department):
    if department == "all":
        existing_files = list(db.uploads.find())
    else:
        existing_files = list(db.uploads.find({"department": department}))
    
    combined_text = ""
    for file in existing_files:
        combined_text += file["content"] + "\n"
    return combined_text

# Split text into chunks
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

# Create vector store from text chunks
def get_vector_store(text_chunks, api_key):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=api_key)
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")
    return vector_store, embeddings

# Create conversational chain
def get_conversational_chain(api_key):
    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
    provided context just say, "answer is not available in the context", don't provide the wrong answer\n\n
    Context:\n {context}?\n
    Question: \n{question}\n
    Answer:
    """
    model = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.3, google_api_key=api_key)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

# Check if a similar query exists in the database for the specific department
def find_similar_query(db, user_question, embeddings, department, threshold=0.8):
    # Get stored queries for the department
    if department == "all":
        stored_queries = list(db.queries.find())
    else:
        stored_queries = list(db.queries.find({"department": department}))
    
    if not stored_queries:
        return None
    
    # Compute embeddings for the user question
    user_embedding = embeddings.embed_query(user_question)
    
    # Compare with stored queries
    for query in stored_queries:
        stored_embedding = query.get("embedding")
        if stored_embedding:
            similarity = cosine_similarity([user_embedding], [stored_embedding])[0][0]
            if similarity >= threshold:
                return query["response"]  # Return the stored answer
    
    return None  # No similar query found

# Process user query
def process_user_query(user_question, api_key, department):
    db = initialize_db()
    
    # Load existing files based on department
    existing_text = load_existing_files(db, department)
    
    # Create text chunks and vector store
    text_chunks = get_text_chunks(existing_text)
    vector_store, embeddings = get_vector_store(text_chunks, api_key)
    
    # Check if a similar query exists for this department
    similar_answer = find_similar_query(db, user_question, embeddings, department)
    
    if similar_answer:
        return similar_answer, True, None
    
    # No similar answer found, query the model
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)
    chain = get_conversational_chain(api_key)
    response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
    response_output = response['output_text']
    
    # Get embedding and handle different return types
    user_embedding = embeddings.embed_query(user_question)
    embedding_list = user_embedding.tolist() if hasattr(user_embedding, 'tolist') else list(user_embedding)
    
    # Check if answer is not available
    if "answer is not available in the context" in response_output.lower():
        # Store unanswered query with department info
        unanswered_query = {
            "user_question": user_question,
            "embedding": embedding_list,
            "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            "status": "pending",
            "department": department
        }
        db.unanswered_queries.insert_one(unanswered_query)
        return response_output, False, "unanswered"
    
    # Store the query in the database with department info
    query_data = {
        "user_question": user_question,
        "response": response_output,
        "embedding": embedding_list,
        "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        "ai_generated": True,
        "department": department
    }
    db.queries.insert_one(query_data)
    
    return response_output, False, None

def main():
    st.set_page_config(
        page_title="User Knowledge Base Portal",
        page_icon="🔍",
        layout="wide"
    )
    
    # Check if user has API key
    if 'api_key' not in st.session_state or not st.session_state.api_key:
        st.warning("Please enter your Google API Key on the home page.")
        if st.button("Return to Home"):
            st.switch_page("Home.py")
        return
    
    # Initialize database
    db = initialize_db()
    
    # Get user's selected department
    if 'user_department' not in st.session_state:
        st.session_state.user_department = "all"
    
    user_department = st.session_state.user_department
    
    # Display title with department context
    if user_department == "all":
        st.title("User Knowledge Base Portal - All Departments")
    else:
        st.title(f"User Knowledge Base Portal - {user_department} Department")
    
    st.write("Ask questions based on the company's knowledge base")
    
    # Allow users to change department after coming to the portal
    departments = get_departments(db)
    dept_options = ["All Departments"] + [dept["name"] for dept in departments]
    selected_dept = st.selectbox("Select Department to Query:", dept_options, 
                                index=dept_options.index(user_department) if user_department != "all" else 0)
    
    if selected_dept != "All Departments":
        st.session_state.user_department = selected_dept
    else:
        st.session_state.user_department = "all"
    
    # Update the working department
    working_department = st.session_state.user_department
    
    # Initialize session state for conversation history
    if 'conversation_history' not in st.session_state:
        st.session_state.conversation_history = []
    
    # Sidebar with options
    with st.sidebar:
        st.title("Options")
        
        if st.button("Return to Home"):
            st.switch_page("Home.py")
            
        if st.button("Clear Chat History"):
            st.session_state.conversation_history = []
            st.rerun()
        
        # Download conversation history
        if len(st.session_state.conversation_history) > 0:
            df = pd.DataFrame(st.session_state.conversation_history, 
                             columns=["Question", "Answer", "Timestamp", "From Knowledge Base", "Department"])
            csv = df.to_csv(index=False)
            b64 = base64.b64encode(csv.encode()).decode()
            href = f'<a href="data:file/csv;base64,{b64}" download="conversation_history.csv"><button>Download Chat History</button></a>'
            st.markdown(href, unsafe_allow_html=True)
    
    # User query input
    user_question = st.text_input("Ask a question:")
    
    if user_question:
        # Process the query for the selected department
        response, from_kb, query_type = process_user_query(user_question, st.session_state.api_key, working_department)
        
        # Add to conversation history with department info
        st.session_state.conversation_history.append(
            (user_question, response, datetime.now().strftime('%Y-%m-%d %H:%M:%S'), from_kb, working_department)
        )
    
    # Display conversation history
    if st.session_state.conversation_history:
        st.subheader("Conversation History")
        for i, (question, answer, timestamp, from_kb, dept) in enumerate(reversed(st.session_state.conversation_history)):
            # Create expander for each Q&A pair
            source_label = "Retrieved from Knowledge Base" if from_kb else "Generated by AI"
            dept_label = f" - {dept}" if dept != "all" else " - All Departments"
            with st.expander(f"Q: {question} ({source_label}{dept_label})"):
                st.info(f"**Question:** {question}")
                st.success(f"**Answer:** {answer}")
                st.caption(f"Time: {timestamp}")
    
    # Display current query and response
    if user_question:
        st.subheader("Your Question")
        st.info(user_question)
        
        st.subheader("Response")
        st.success(response)
        
        # Show notification for unanswered queries
        if query_type == "unanswered":
            st.warning("This question couldn't be fully answered from the knowledge base. It has been sent to department administrators for a more complete answer.")

if __name__ == "__main__":
    main()
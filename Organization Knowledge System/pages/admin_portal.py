# pages/admin_portal.py
import streamlit as st
import pandas as pd
from pymongo import MongoClient
import json
from datetime import datetime
from PyPDF2 import PdfReader
from docx import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import os
import shutil
import csv
import io

# Initialize MongoDB connection
def get_mongo_client():
    client = MongoClient("mongodb://localhost:27017/")
    return client

def initialize_db():
    client = get_mongo_client()
    db = client["knowledge_base"]
    return db

# Check authentication
def check_auth():
    if 'authenticated' not in st.session_state or not st.session_state.authenticated:
        st.warning("Please log in as admin first.")
        if st.button("Return to Login"):
            st.switch_page("Home.py")
        st.stop()

# Get departments
def get_departments(db):
    return list(db.departments.find())

# Extract text from PDFs
def extract_text_from_pdf(pdf):
    text = ""
    pdf_reader = PdfReader(pdf)
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

# Extract text from JSON
def extract_text_from_json(json_file):
    data = json.load(json_file)
    return json.dumps(data, indent=2)

# Extract text from Excel
def extract_text_from_excel(excel_file):
    df = pd.read_excel(excel_file)
    return df.to_string()

# Extract text from CSV
def extract_text_from_csv(csv_file):
    # Reset file pointer to the beginning
    csv_file.seek(0)
    # Use pandas to read CSV with automatic delimiter detection
    try:
        df = pd.read_csv(csv_file)
    except:
        # Try with different delimiter if default fails
        csv_file.seek(0)
        df = pd.read_csv(csv_file, sep=';')
    
    return df.to_string()

# Extract text from DOC/DOCX
def extract_text_from_doc(doc_file):
    doc = Document(doc_file)
    text = ""
    for para in doc.paragraphs:
        text += para.text + "\n"
    return text

# Create department folder if it doesn't exist
def ensure_department_folder(department):
    folder_path = f"department_data/{department}"
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    return folder_path

# Process uploaded files and store in MongoDB and local folder
def process_uploaded_files(uploaded_files, db, department):
    results = []
    
    # Create department folder if it doesn't exist
    department_folder = ensure_department_folder(department)
    
    for uploaded_file in uploaded_files:
        file_details = {
            "filename": uploaded_file.name,
            "file_type": uploaded_file.type,
            "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            "source": "uploaded_by_admin",
            "department": department
        }
        
        # Save file locally in department folder
        with open(f"{department_folder}/{uploaded_file.name}", "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        # Extract text based on file type
        try:
            if uploaded_file.type == "application/pdf":
                text = extract_text_from_pdf(uploaded_file)
            elif uploaded_file.type == "application/json":
                text = extract_text_from_json(uploaded_file)
            elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet" or uploaded_file.type == "application/vnd.ms-excel":
                text = extract_text_from_excel(uploaded_file)
            elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
                text = extract_text_from_doc(uploaded_file)
            elif uploaded_file.type == "text/csv" or uploaded_file.name.endswith('.csv'):
                text = extract_text_from_csv(uploaded_file)
            else:
                results.append((uploaded_file.name, "Error", f"Unsupported file type: {uploaded_file.type}"))
                continue
            
            # Store file data in MongoDB
            file_details["content"] = text
            db.uploads.insert_one(file_details)
            results.append((uploaded_file.name, "Success", f"File processed and stored successfully in {department} department"))
        
        except Exception as e:
            results.append((uploaded_file.name, "Error", str(e)))
    
    return results

# Get unanswered queries from the database based on department
def get_unanswered_queries(db, department):
    if department == "all":
        return list(db.unanswered_queries.find({"status": "pending"}))
    else:
        return list(db.unanswered_queries.find({"status": "pending", "department": department}))

# Update query with admin's answer
def update_query_with_answer(db, query_id, admin_answer, api_key):
    # Get the query from unanswered_queries
    query_doc = db.unanswered_queries.find_one({"_id": query_id})
    
    if not query_doc:
        return False, "Query not found"
    
    try:
        # Create embeddings
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=api_key)
        
        # Add to queries collection
        query_data = {
            "user_question": query_doc["user_question"],
            "response": admin_answer,
            "embedding": query_doc["embedding"],
            "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            "answered_by": "admin",
            "original_query_timestamp": query_doc["timestamp"],
            "department": query_doc.get("department", "all")
        }
        db.queries.insert_one(query_data)
        
        # Update status in unanswered_queries
        db.unanswered_queries.update_one(
            {"_id": query_id},
            {"$set": {"status": "answered", "admin_answer": admin_answer}}
        )
        
        return True, "Answer added successfully"
    
    except Exception as e:
        return False, str(e)

def main():
    st.set_page_config(
        page_title="Admin Knowledge Base Portal",
        page_icon="🔧",
        layout="wide"
    )
    
    # Check authentication
    check_auth()
    
    # Get API key from session state
    if 'api_key' not in st.session_state or not st.session_state.api_key:
        st.warning("Please enter your Google API Key")
        st.session_state.api_key = st.text_input("Enter your Google API Key:", type="password")
    
    # Initialize database
    db = initialize_db()
    
    # Get admin's department access
    admin_department = st.session_state.admin_department
    
    st.title(f"Admin Knowledge Base Portal - {admin_department if admin_department != 'all' else 'All Departments'}")
    
    # Department selection for super admin
    if admin_department == "all":
        departments = get_departments(db)
        dept_options = ["All Departments"] + [dept["name"] for dept in departments]
        selected_dept = st.selectbox("Select Department to Manage:", dept_options)
        
        if selected_dept != "All Departments":
            working_department = selected_dept
        else:
            working_department = "all"
    else:
        working_department = admin_department
        st.info(f"You have access to manage the {working_department} department only")
    
    # Create tabs for different admin functions
    tab1, tab2, tab3 = st.tabs(["Manage Documents", "Answer Queries", "View Statistics"])
    
    # Tab 1: Manage Documents
    with tab1:
        st.header(f"Upload Documents to {working_department if working_department != 'all' else 'All Departments'} Knowledge Base")
        
        if working_department == "all":
            # If admin is working with all departments, let them select a specific department for upload
            upload_departments = [dept["name"] for dept in get_departments(db)]
            upload_department = st.selectbox("Select Department for this Upload:", upload_departments)
        else:
            upload_department = working_department
        
        uploaded_files = st.file_uploader("Upload PDF, JSON, Excel, CSV, or DOC files", 
                                        accept_multiple_files=True,
                                        type=["pdf", "json", "xlsx", "docx", "xls", "csv"])
        
        if st.button("Process Files") and uploaded_files:
            with st.spinner(f"Processing files for {upload_department} department..."):
                results = process_uploaded_files(uploaded_files, db, upload_department)
                
                # Display results
                st.subheader("Processing Results")
                for filename, status, message in results:
                    if status == "Success":
                        st.success(f"{filename}: {message}")
                    else:
                        st.error(f"{filename}: {message}")
        
        # Display existing files
        st.subheader("Existing Documents in Knowledge Base")
        
        # Filter files based on department access
        if working_department == "all":
            existing_files = list(db.uploads.find({}, {"filename": 1, "timestamp": 1, "source": 1, "department": 1}))
        else:
            existing_files = list(db.uploads.find({"department": working_department}, 
                                                  {"filename": 1, "timestamp": 1, "source": 1, "department": 1}))
        
        if existing_files:
            file_data = []
            for file in existing_files:
                file_data.append({
                    "Filename": file["filename"],
                    "Department": file.get("department", "Unassigned"),
                    "Uploaded On": file["timestamp"],
                    "Source": file["source"]
                })
            
            st.dataframe(pd.DataFrame(file_data))
        else:
            st.info(f"No documents in the {working_department if working_department != 'all' else 'knowledge base'} yet.")
    
    # Tab 2: Answer Queries
    with tab2:
        st.header(f"Unanswered User Queries - {working_department if working_department != 'all' else 'All Departments'}")
        
        # Get unanswered queries
        unanswered_queries = get_unanswered_queries(db, working_department)
        
        if not unanswered_queries:
            st.info(f"No unanswered queries for {working_department if working_department != 'all' else 'any department'} at the moment.")
        else:
            st.write(f"Found {len(unanswered_queries)} unanswered queries.")
            
            # Create an expander for each query
            for i, query in enumerate(unanswered_queries):
                dept_label = f" - {query.get('department', 'Unassigned')}" if working_department == "all" else ""
                with st.expander(f"Query {i+1}{dept_label}: {query['user_question'][:50]}..."):
                    st.write(f"**Full Question:** {query['user_question']}")
                    st.write(f"**Timestamp:** {query['timestamp']}")
                    if working_department == "all":
                        st.write(f"**Department:** {query.get('department', 'Unassigned')}")
                    
                    # Form for admin to answer
                    with st.form(key=f"answer_form_{i}"):
                        admin_answer = st.text_area("Your Answer:", key=f"answer_{i}", height=150)
                        submit_button = st.form_submit_button("Submit Answer")
                        
                        if submit_button and admin_answer:
                            success, message = update_query_with_answer(
                                db, query["_id"], admin_answer, st.session_state.api_key
                            )
                            
                            if success:
                                st.success(message)
                                st.rerun()
                            else:
                                st.error(f"Error: {message}")
    
    # Tab 3: View Statistics
    with tab3:
        st.header(f"Knowledge Base Statistics - {working_department if working_department != 'all' else 'All Departments'}")
        
        # Get counts based on department
        if working_department == "all":
            doc_count = db.uploads.count_documents({})
            query_count = db.queries.count_documents({})
            unanswered_count = db.unanswered_queries.count_documents({"status": "pending"})
            admin_answered = db.queries.count_documents({"answered_by": "admin"})
        else:
            doc_count = db.uploads.count_documents({"department": working_department})
            query_count = db.queries.count_documents({"department": working_department})
            unanswered_count = db.unanswered_queries.count_documents({"status": "pending", "department": working_department})
            admin_answered = db.queries.count_documents({"answered_by": "admin", "department": working_department})
        
        # Display counts
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Documents", doc_count)
        
        with col2:
            st.metric("Total Answered Queries", query_count)
        
        with col3:
            st.metric("Pending Queries", unanswered_count)
        
        with col4:
            st.metric("Admin Answered", admin_answered)
        
        # Show recent queries
        st.subheader("Recent Answered Queries")
        
        # Filter queries based on department
        if working_department == "all":
            recent_queries = list(db.queries.find().sort("timestamp", -1).limit(10))
        else:
            recent_queries = list(db.queries.find({"department": working_department}).sort("timestamp", -1).limit(10))
        
        if recent_queries:
            query_data = []
            for query in recent_queries:
                query_data.append({
                    "Question": query["user_question"][:50] + "..." if len(query["user_question"]) > 50 else query["user_question"],
                    "Department": query.get("department", "Unassigned"),
                    "Answered By": query.get("answered_by", "AI"),
                    "Timestamp": query["timestamp"]
                })
            
            st.dataframe(pd.DataFrame(query_data))
        else:
            st.info("No answered queries yet.")
    
    # Logout button in sidebar
    with st.sidebar:
        st.title("Admin Options")
        
        if st.button("Logout"):
            st.session_state.authenticated = False
            st.switch_page("Home.py")
        
        # Department management for super admin
        if admin_department == "all":
            st.divider()
            st.subheader("Department Management")
            
            with st.expander("Add New Department"):
                new_dept_name = st.text_input("Department Name:")
                new_dept_desc = st.text_input("Department Description:")
                
                if st.button("Add Department"):
                    if new_dept_name:
                        # Check if department already exists
                        existing_dept = db.departments.find_one({"name": new_dept_name})
                        if existing_dept:
                            st.error("Department already exists")
                        else:
                            db.departments.insert_one({
                                "name": new_dept_name,
                                "description": new_dept_desc
                            })
                            # Create department folder
                            ensure_department_folder(new_dept_name)
                            st.success(f"Department {new_dept_name} added successfully")
                            st.rerun()

if __name__ == "__main__":
    main()
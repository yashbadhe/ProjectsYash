# Home.py (Main entry point with department support)
import streamlit as st
import pandas as pd
from pymongo import MongoClient
import os
import base64

# Initialize MongoDB connection
def get_mongo_client():
    # Replace with your MongoDB connection string
    client = MongoClient("mongodb://localhost:27017/")
    return client

def initialize_db():
    client = get_mongo_client()
    db = client["knowledge_base"]
    
    # Create collections if they don't exist
    if "uploads" not in db.list_collection_names():
        db.create_collection("uploads")
    
    if "queries" not in db.list_collection_names():
        db.create_collection("queries")
        
    if "unanswered_queries" not in db.list_collection_names():
        db.create_collection("unanswered_queries")
    
    if "departments" not in db.list_collection_names():
        departments_collection = db.create_collection("departments")
        # Add default departments
        default_departments = [
            {"name": "HR", "description": "Human Resources Department"},
            {"name": "Finance", "description": "Finance Department"},
            {"name": "IT", "description": "Information Technology Department"},
            {"name": "Operations", "description": "Operations Department"}
        ]
        departments_collection.insert_many(default_departments)
    
    if "admin_departments" not in db.list_collection_names():
        db.create_collection("admin_departments")
        
    return db

# Get all departments
def get_departments(db):
    return list(db.departments.find())

# Simple authentication with department access
def authenticate(username, password, db):
    # In a real app, use proper authentication with hashed passwords
    if username == "admin" and password == "admin123":
        # Super admin has access to all departments
        return True, "all"
    
    # Check for department admin
    admin_dept = db.admin_departments.find_one({"username": username, "password": password})
    if admin_dept:
        return True, admin_dept["department"]
    
    return False, None

def main():
    st.set_page_config(
        page_title="Knowledge Base App",
        page_icon="📚",
        layout="wide"
    )
    
    # Initialize database
    db = initialize_db()
    
    # Session state initialization
    if 'authenticated' not in st.session_state:
        st.session_state.authenticated = False
    
    if 'api_key' not in st.session_state:
        st.session_state.api_key = None
    
    if 'admin_department' not in st.session_state:
        st.session_state.admin_department = None
    
    # App title and description
    st.title("📚 Department-wise Knowledge Base System")
    
    # Create tabs for user and admin login
    tab1, tab2 = st.tabs(["User Access", "Admin Login"])
    
    with tab1:
        st.header("User Access")
        st.write("As a user, you can query the knowledge base and ask questions based on the existing documents.")
        st.session_state.api_key = "SESSION API" #API
        
        # Ask for API key if not already provided
        if not st.session_state.api_key:
            st.session_state.api_key = st.text_input("Enter your Google API Key:", type="password")
            st.markdown("Click [here](https://ai.google.dev/) to get an API key.")
        
        # Departments selection for users
        departments = get_departments(db)
        dept_options = ["All Departments"] + [dept["name"] for dept in departments]
        selected_dept = st.selectbox("Select Department to Query:", dept_options)
        
        if selected_dept != "All Departments":
            st.session_state.user_department = selected_dept
        else:
            st.session_state.user_department = "all"
            
        if st.session_state.api_key:
            if st.button("Go to User Portal"):
                st.switch_page("pages/user_portal.py")
    
    with tab2:
        st.header("Admin Login")
        st.write("Admin access allows you to manage documents and answer user queries.")
        
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        
        if st.button("Login"):
            auth_result, department = authenticate(username, password, db)
            if auth_result:
                st.session_state.authenticated = True
                st.session_state.admin_department = department
                st.success(f"Login successful! You have access to {department if department != 'all' else 'all departments'}")
                st.switch_page("pages/admin_portal.py")
            else:
                st.error("Invalid username or password")
        
        st.divider()
        
        # Department admin registration (in a real app, this would be restricted to super admin)
        if st.checkbox("Register Department Admin"):
            st.subheader("Register New Department Admin")
            
            new_admin_dept = st.selectbox("Department:", [dept["name"] for dept in get_departments(db)])
            new_admin_username = st.text_input("Admin Username:")
            new_admin_password = st.text_input("Admin Password:", type="password")
            
            if st.button("Register Department Admin"):
                # Check if username already exists
                existing_admin = db.admin_departments.find_one({"username": new_admin_username})
                if existing_admin:
                    st.error("Username already exists")
                else:
                    db.admin_departments.insert_one({
                        "username": new_admin_username,
                        "password": new_admin_password,  # In a real app, hash this password
                        "department": new_admin_dept
                    })
                    st.success(f"Admin for {new_admin_dept} department registered successfully")

# Display supported file types in the sidebar
    with st.sidebar:
        st.subheader("Supported File Types")
        st.write("The knowledge base system supports the following file formats:")
        st.write("• PDF (.pdf)")
        st.write("• Word (.docx)")
        st.write("• Excel (.xlsx, .xls)")
        st.write("• CSV (.csv)")
        st.write("• JSON (.json)")
        
        st.caption("Files will be processed and stored in the knowledge base for querying.")

if __name__ == "__main__":
    main()
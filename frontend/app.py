import os
import requests
import streamlit as st

from dotenv import load_dotenv

# load env
load_dotenv()

# API URL
API_BASE_URL = f"http://{os.getenv('API_HOST', 'localhost')}:{os.getenv('API_PORT', 8000)}"

# streamlit page config
st.set_page_config(
    page_title="Chatbot",
    layout="wide",
    initial_sidebar_state="expanded"
)

class APIInterface:
    def __init__(self):
        self.api_base = API_BASE_URL

    # upload document
    def upload_document(self, file_data, filename):
        try:
            files = {'file': (filename, file_data, 'application/octet-stream')}
            response = requests.post(f"{self.api_base}/upload-document", files=files)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            st.error(f"Error uploading file: {str(e)}")
            return None

    # query documents 
    def query_documents(self, question, max_results=3):
        try:
            payload = {
                "question": question,
                "max_results": max_results
            }
            response = requests.post(f"{self.api_base}/query", json=payload)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            st.error(f"Error querying documents: {str(e)}")
            return None

    # list of documents
    def get_documents(self):
        try:
            response = requests.get(f"{self.api_base}/documents")
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            st.error(f"Error fetching documents: {str(e)}")
            return None

    # delete document
    def delete_document(self, doc_id):
        try:
            response = requests.delete(f"{self.api_base}/documents/{doc_id}")
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            st.error(f"Error deleting document: {str(e)}")
            return None

    # health check API
    def check_api_health(self):
        try:
            response = requests.get(f"{self.api_base}/health", timeout=5)
            return response.status_code == 200
        except:
            return False


# init session state
def initialize_session_state():
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    if 'documents' not in st.session_state:
        st.session_state.documents = []
    if 'api_interface' not in st.session_state:
        st.session_state.api_interface = APIInterface()

# main function
def main():
    initialize_session_state()
    api = st.session_state.api_interface

    # title
    st.title("Chatbot")
    st.markdown("Upload document and ask questions to get answers form document content.")

    # check backend connection
    api_connected = api.check_api_health()
    
    if not api_connected:
        st.error("Cannot connect to the server. Make sure the backend is running.")
        if st.button("Retry Connection"):
            st.rerun()
        return
    
    # sidebar
    with st.sidebar:
        st.header("Document Management")
        
        # upload file section
        st.subheader("Upload Document")
        uploaded_file = st.file_uploader(
            "Choose a file",
            type=['pdf', 'txt'],
            help="Upload PDF or TXT files"
        )

        if uploaded_file is not None:
            if st.button("Upload Document", type="primary"):
                with st.spinner("Processing document..."):
                    file_data = uploaded_file.getvalue()
                    result = api.upload_document(file_data, uploaded_file.name)
                    
                    if result:
                        st.success(f"Document '{uploaded_file.name}' uploaded successfully!")
                        st.rerun()

        # document list section
        st.subheader("Uploaded Documents")
        
        # refresh button
        if st.button("Refresh", key="refresh_docs"):
            st.rerun()

        docs_data = api.get_documents()
        if docs_data and docs_data.get('documents'):
            st.session_state.documents = docs_data['documents']
            
            st.info(f"Total documents: {docs_data.get('total', 0)}")
            
            for doc in st.session_state.documents:
                with st.expander(f"{doc['filename']}", expanded=False):
                    if st.button(f"Delete", key=f"delete_{doc['id']}"):
                        with st.spinner("Deleting document..."):
                            delete_result = api.delete_document(doc['id'])
                            if delete_result:
                                st.success("Document deleted!")
                                st.rerun()
        else:
            st.info("No documents uploaded yet.")

    # chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # user input
    if prompt := st.chat_input("Ask a question..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # get AI response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = api.query_documents(prompt)
                
                if response:
                    st.markdown(response['answer'])
                    
                    # store message
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": response['answer'],
                    })

                else:
                    error_msg = "Sorry, I couldn't process your question. Please try again."
                    st.error(error_msg)
                    st.session_state.messages.append({"role": "assistant", "content": error_msg})

if __name__ == "__main__":
    main()
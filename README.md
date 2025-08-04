# Chatbot
This project is a chatbot that answers questions based on documents uploaded by the user. 

## Technical stack
Backend: FastAPI  
Frontend: Streamlit

## Prerequisites
Python 3.8 or higher

## Create virtual environment
python -m venv venv  

On Windows  
venv\Scripts\activate  

On macOS or Linux  
source venv/bin/activate

## Install dependencies
Install all the required Python packages from the requirements.txt file.  
pip install -r requirements.txt

## NOTE: Make sure to add .env file. take reference from example.env file.
## NOTE: Make sure to create vector_store directory.

## Start the Backend API
Navigate into the backend directory and start the backend server.  

cd backend  
python run_backend.py

## Start the Frontend
Open a new terminal, navigate into the frontend directory, and run the frontend application.  

cd frontend  
python run_frontend.py

import os
import uvicorn

from datetime import datetime
from dotenv import load_dotenv

from fastapi import FastAPI, UploadFile, File, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware

from backend.models import (
    DocumentUploadResponse, 
    QueryRequest, 
    QueryResponse, 
    DocumentList, 
    DocumentInfo,
    DeleteResponse
)
from backend.rag_system import RAGSystem

# load env
load_dotenv()

# FastAPI app
app = FastAPI(
    title="Chatbot API",
    description="API for document upload and question answering using RAG",
    version="1"
)

# middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# init RAG system
def get_rag_system():
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        raise HTTPException(status_code=500, detail="OpenAI API key not fount")
    
    vector_store_path = os.getenv("VECTOR_STORE_PATH", "./vector_store")
    return RAGSystem(openai_api_key, vector_store_path)

@app.get("/")
async def root():
    return {"message": "Backend API is running", "version": "1"}

# upload a document
@app.post("/upload-document", response_model=DocumentUploadResponse)
async def upload_document(
    file: UploadFile = File(...),
    rag_system: RAGSystem = Depends(get_rag_system)
):
    try:
        # file type validation
        if not str(file.filename).lower().endswith(('.pdf', '.txt')):
            raise HTTPException(status_code=400, detail="Only PDF and TXT files are supported")
        
        file_content = await file.read()
        if len(file_content) == 0:
            raise HTTPException(status_code=400, detail="Empty file uploaded")

        # process document
        doc_id = rag_system.add_document(file_content, str(file.filename))

        return DocumentUploadResponse(
            id=doc_id,
            filename=str(file.filename),
            status="success",
            message="Document uploaded and processed successfully",
            timestamp=datetime.now()
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# query documents 
@app.post("/query", response_model=QueryResponse)
async def query_documents(
    request: QueryRequest,
    rag_system: RAGSystem = Depends(get_rag_system)
):
    try:
        if not request.question.strip():
            raise HTTPException(status_code=400, detail="Question cannot be empty")

        # check for existing documents
        if rag_system.get_document_count() == 0:
            return QueryResponse(answer="No documents uploaded yet.",)

        # generate and return answer
        result = rag_system.generate_answer(request.question, int(str(request.max_results)))
        return QueryResponse(answer=result['answer'])
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# list of documents
@app.get("/documents", response_model=DocumentList)
async def get_documents(rag_system: RAGSystem = Depends(get_rag_system)):
    try:
        documents = rag_system.get_documents()
        document_list = [
            DocumentInfo(id=doc['id'], filename=doc['filename'])
            for doc in documents
        ]

        return DocumentList(documents=document_list, total=len(document_list))
    except Exception as e:
        print(str(e))
        raise HTTPException(status_code=500, detail=str(e))

# delete document
@app.delete("/documents/{doc_id}", response_model=DeleteResponse)
async def delete_document(
    doc_id: str,
    rag_system: RAGSystem = Depends(get_rag_system)
):
    try:
        success = rag_system.delete_document(doc_id)
        if not success:
            raise HTTPException(status_code=404, detail="Document not found")
        
        return DeleteResponse(message="Document deleted successfully", deleted_id=doc_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# health check endpoint
@app.get("/health")
async def health_check():
    try:
        return {
            "status": "healthy",
            "timestamp": datetime.now(),
            "api_version": "1.0.0"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Health check failed: {str(e)}")


if __name__ == "__main__":
    host = os.getenv("API_HOST", "localhost")
    port = int(os.getenv("API_PORT", 8000))
    uvicorn.run(app, host=host, port=port)
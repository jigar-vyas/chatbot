from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime

class DocumentUploadResponse(BaseModel):
    id: str
    filename: str
    status: str
    message: str
    timestamp: datetime

class QueryRequest(BaseModel):
    question: str
    max_results: Optional[int] = 3

class QueryResponse(BaseModel):
    answer: str

class DocumentInfo(BaseModel):
    id: str
    filename: str

class DocumentList(BaseModel):
    documents: List[DocumentInfo]
    total: int

class DeleteResponse(BaseModel):
    message: str
    deleted_id: str
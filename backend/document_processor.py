import os
import uuid
import json
from datetime import datetime
from typing import List, Dict, Optional
import chromadb
from chromadb.config import Settings
from openai import OpenAI
import PyPDF2
import io
import numpy as np
from chromadb.errors import NotFoundError
from dotenv import load_dotenv

# load env
load_dotenv()

class DocumentProcessor:

    def __init__(self, vector_store_path: str = "./vector_store", openai_api_key: str = None):
        self.vector_store_path = vector_store_path
        self.collection_name = "documents"
        
        # OpenAI embeddings
        if not openai_api_key:
            openai_api_key = os.environ["OPENAI_API_KEY"]
        
        if not openai_api_key:
            raise ValueError("OpenAI API key is required for embeddings")
        
        self.openai_client = OpenAI(api_key=openai_api_key)
        self.embedding_model = "text-embedding-3-small" 
        
        self._init_chromadb()
        
        self.metadata_file = os.path.join(vector_store_path, "documents_metadata.json")
        self._ensure_metadata_file()


    # init ChromaDB
    def _init_chromadb(self):
        
        os.makedirs(self.vector_store_path, exist_ok=True)

        self.client = chromadb.PersistentClient(
            path=self.vector_store_path,
            settings=Settings(anonymized_telemetry=False)
        )
        
        base_name = self.collection_name
        openai_name = f"{base_name}_openai"

        try:
            self.collection = self.client.get_collection(name=openai_name)
        except NotFoundError:
            print(f"No '{openai_name}' found—creating a new one.")

            try:
                _ = self.client.get_collection(name=base_name)
                self.client.delete_collection(name=base_name)
            except NotFoundError:
                pass

            #  collection
            self.collection = self.client.create_collection(
                name=openai_name,
                metadata={
                    "hnsw:space": "cosine",
                    "embedding_model": self.embedding_model,
                    "embedding_dimension": 1536,
                    "created_at": datetime.now().isoformat()
                }
            )
            
            # clear old metadata
            self._clear_metadata()


    # get embedding from OpenAI
    def _get_openai_embedding(self, text: str) -> List[float]:
        try:
            text = text.replace("\n", " ").strip()
            if not text:
                return []
            
            # embedding using OpenAI
            response = self.openai_client.embeddings.create(
                input=text,
                model=self.embedding_model
            )
            
            embedding = response.data[0].embedding
            return embedding
            
        except Exception as e:
            print(f"Error getting OpenAI embedding: {str(e)}")
            raise Exception(f"Error getting OpenAI embedding: {str(e)}")


    # get embeddings for multiple texts in batch
    def _get_batch_openai_embeddings(self, texts: List[str]) -> List[List[float]]:
        try:
            cleaned_texts = [text.replace("\n", " ").strip() for text in texts if text.strip()]
            if not cleaned_texts:
                return []
            
            # embeddings using OpenAI
            response = self.openai_client.embeddings.create(
                input=cleaned_texts,
                model=self.embedding_model
            )
            
            embeddings = [data.embedding for data in response.data]
            return embeddings
            
        except Exception as e:
            print(f"Error getting embeddings: {str(e)}")
            raise Exception(f"Error getting embeddings: {str(e)}")


    # clear metadata files
    def _clear_metadata(self):
        try:
            with open(self.metadata_file, 'w') as f:
                json.dump({}, f)
        except Exception as e:
            print(f"Error in deleting metadata file: {e}")

    
    # check metadata file exists or not
    def _ensure_metadata_file(self):
        if not os.path.exists(self.metadata_file):
            with open(self.metadata_file, 'w') as f:
                json.dump({}, f)

    
    # load document metadata
    def _load_metadata(self) -> Dict:
        try:
            with open(self.metadata_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error in loading metadata: {e}")
            return {}


    # save document metadata
    def _save_metadata(self, metadata: Dict):
        try:
            with open(self.metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2, default=str)
        except Exception as e:
            print(f"Error in saveing metadata: {e}")


    # extension based text extraction
    def _extract_text_from_file(self, file_content: bytes, filename: str) -> str:
        file_ext = filename.lower().split('.')[-1]
        
        if file_ext == 'txt':
            try:
                return file_content.decode('utf-8', errors='ignore')
            except Exception as e:
                for encoding in ['latin-1', 'cp1252', 'iso-8859-1']:
                    try:
                        return file_content.decode(encoding, errors='ignore')
                    except:
                        continue
                raise Exception(f"Error decoding text file: {str(e)}")
        
        elif file_ext == 'pdf':
            try:
                pdf_file = io.BytesIO(file_content)
                pdf_reader = PyPDF2.PdfReader(pdf_file)
                text = ""
                for page_num, page in enumerate(pdf_reader.pages):
                    try:
                        page_text = page.extract_text()
                        if page_text:
                            text += f"\n--- Page {page_num + 1} ---\n{page_text}\n"
                    except Exception as e:
                        print(f"Error in extracting text from page {page_num + 1}: {e}")
                        continue
                
                if not text.strip():
                    raise Exception("No text could be extracted from PDF")
                
                return text
            except Exception as e:
                raise Exception(f"Error in reading PDF: {str(e)}")
        
        else:
            raise Exception(f"Only PDF and TXT files are supported.")

    
    # split text into chunks
    def _chunk_text(self, text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
        if len(text) <= chunk_size:
            return [text.strip()] if text.strip() else []
        
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + chunk_size
            if end < len(text):
                # check sentence boundary (period, exclamation, question mark)
                sentence_boundaries = ['.', '!', '?']
                best_boundary = -1
                
                for boundary in sentence_boundaries:
                    boundary_pos = text.rfind(boundary, start, end)
                    if boundary_pos > start + chunk_size // 2:
                        best_boundary = max(best_boundary, boundary_pos + 1)
                
                if best_boundary > 0:
                    end = best_boundary
                else:
                    # check word boundary
                    word_end = text.rfind(' ', start, end)
                    if word_end > start + chunk_size // 2:
                        end = word_end
            
            chunk = text[start:end].strip()
            if chunk and len(chunk) > 50: 
                chunks.append(chunk)
            
            start = end - overlap
            
        return chunks


    # process and store document
    def process_document(self, file_content: bytes, filename: str) -> str:
        try:
            doc_id = str(uuid.uuid4())
            text = self._extract_text_from_file(file_content, filename)
            if not text.strip():
                raise Exception("No text content found")
            
            # chunk text
            chunks = self._chunk_text(text)
            if not chunks:
                raise Exception("No text chunks created from the document")
            
            # generate embeddings for all chunks
            embeddings = self._get_batch_openai_embeddings(chunks)
            if len(embeddings) != len(chunks):
                raise Exception(f"chunks ({len(chunks)}) and embeddings ({len(embeddings)}) not matched")
            
            # prepare data for ChromaDB
            ids = []
            documents = []
            metadatas = []
            
            for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
                chunk_id = f"{doc_id}_{i}"
                ids.append(chunk_id)
                documents.append(chunk)
                metadatas.append({
                    "doc_id": doc_id,
                    "filename": filename,
                    "chunk_index": i,
                    "chunk_id": chunk_id
                })
            
            # add to ChromaDB with embeddings
            self.collection.add(
                ids=ids,
                embeddings=embeddings, 
                documents=documents,
                metadatas=metadatas
            )
            
            # update metadata file
            metadata = self._load_metadata()
            metadata[doc_id] = {
                "id": doc_id,
                "filename": filename,
                "upload_time": datetime.now().isoformat(),
                "size": len(file_content),
                "status": "processed",
                "chunk_count": len(chunks),
                "embedding_model": self.embedding_model
            }
            self._save_metadata(metadata)
            return doc_id
            
        except Exception as e:
            print(f"Erro inr processing document {filename}: {str(e)}")
            raise Exception(f"Error in processing document: {str(e)}")

    
    # query documents using semantic search with OpenAI embeddings
    def query_documents(self, query: str, max_results: int = 3) -> List[Dict]:
        try:
            query_embedding = self._get_openai_embedding(query)
            if not query_embedding:
                return []
            
            # query ChromaDB with the embedding
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=max_results,
                include=["documents", "metadatas", "distances"]
            )
            
            if not results['documents'] or not results['documents'][0]:
                print("No documents found")
                return []
            
            # Format results
            sources = []
            for i in range(len(results['documents'][0])):
                # Convert distance to similarity score (ChromaDB uses cosine distance)
                distance = results['distances'][0][i]
                similarity_score = 1.0 - distance  # Convert distance to similarity
                
                source = {
                    "content": results['documents'][0][i],
                    "filename": results['metadatas'][0][i]['filename'],
                    "doc_id": results['metadatas'][0][i]['doc_id'],
                    "chunk_id": results['metadatas'][0][i]['chunk_id'],
                    "similarity_score": round(max(0.0, similarity_score), 3)
                }
                sources.append(source)
            
            return sources
            
        except Exception as e:
            print(f"Error in querying documents: {str(e)}")
            return []

    
    # get all document metadata
    def get_all_documents(self) -> List[Dict]:
        try:
            metadata = self._load_metadata()
            return list(metadata.values())
        except Exception as e:
            print(f"Error in getting documents: {e}")
            return []

    
    # delete document
    def delete_document(self, doc_id: str) -> bool:
        try:
            metadata = self._load_metadata()
            if doc_id not in metadata:
                return False
            
            # delete chunks from ChromaDB
            results = self.collection.get(
                where={"doc_id": doc_id},
                include=["metadatas"]
            )
            if results['ids']:
                self.collection.delete(ids=results['ids'])
            
            # delete from metadata
            doc_filename = metadata[doc_id].get('filename', 'unknown')
            del metadata[doc_id]
            self._save_metadata(metadata)
            return True
            
        except Exception as e:
            print(f"Error in deleting document: {str(e)}")
            return False

    
    # total count of documents
    def get_document_count(self) -> int:
        try:
            metadata = self._load_metadata()
            count = len(metadata)
            return count
        except Exception as e:
            print(f"Error in getting document count: {e}")
            return 0

    
    # embedding model information
    def get_embedding_info(self) -> Dict:
        return {
            "model": self.embedding_model,
            "provider": "OpenAI",
            "dimension": 1536 if "3-small" in self.embedding_model else 3072,
            "max_tokens": 8191
        }
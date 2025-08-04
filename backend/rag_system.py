import os
from typing import List, Dict
from openai import OpenAI
from backend.document_processor import DocumentProcessor

class RAGSystem:

    def __init__(self, openai_api_key: str, vector_store_path: str = "./vector_store"):
        if not openai_api_key or openai_api_key.strip() == "":
            raise ValueError("OpenAI API key cannot be empty")
        
        try:
            self.client = OpenAI(api_key=openai_api_key)
            self._test_openai_connection()
        except Exception as e:
            print(f"Failed to connect OpenAI: {str(e)}")
            raise
        
        try:
            self.document_processor = DocumentProcessor(vector_store_path)
        except Exception as e:
            print(f"Failed to init document processor: {str(e)}")
            raise
        
        self.system_prompt = """You are a helpful assistant that answers questions based ONLY on the provided context documents. 

Instructions:
1. Use ONLY the information from the provided context to answer questions
2. If the context doesn't contain enough information to answer the question, respond with "I do not know" 
3. Do not use your general knowledge - stick strictly to the provided context
4. Be concise and accurate in your responses
5. If you reference specific information, mention which document it came from

Context Documents:
{context}

Question: {question}

Answer:"""

    
    # OpenAI API connection test
    def _test_openai_connection(self):
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-2024-08-06",
                messages=[{"role": "user", "content": "Hello"}],
                max_tokens=5
            )
        except Exception as e:
            print(f"OpenAI API test failed: {str(e)}")
            raise

    
    # generate answer
    def generate_answer(self, question: str, max_results: int = 3) -> Dict:
        try:
            sources = self.document_processor.query_documents(question, max_results)
            if not sources:
                return {'answer': "No documents found."}
            
            # if the similarity scores are too low 
            if all(source['similarity_score'] < 0.3 for source in sources):
                return {'answer': "I do not know. The question doesn't related to the available documents."}

            # create context from retrieved documents
            context = "\n\n".join([
                f"Document: {source['filename']}\nContent: {source['content']}"
                for source in sources
            ])

            # create prompt
            prompt = self.system_prompt.format(context=context, question=question)
            
            # OpenAI API call
            try:
                response = self.client.chat.completions.create(
                    model="gpt-4o-2024-08-06",
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant that answers questions based only on provided context."},
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=500,
                    temperature=0.1
                )
            except Exception as openai_error:
                print(f"OpenAI API call failed: {str(openai_error)}")
                return {'answer': f"Error connecting to OpenAI: {str(openai_error)}"}

            answer = response.choices[0].message.content.strip()
            
            # Check if the model says it doesn't know
            if any(phrase in answer.lower() for phrase in ["i do not know", "i don't know", "cannot answer", "not enough information"]):
                return {'answer': "I do not know. Available documents don't contain information to answer your question."}
            
            return {'answer': answer}
        except Exception as e:
            print(f"Error in generating answer: {str(e)}")
            return {'answer': f"Error in generating answer: {str(e)}"}


    # add document
    def add_document(self, file_content: bytes, filename: str) -> str:
        try:
            doc_id = self.document_processor.process_document(file_content, filename)
            return doc_id
        except Exception as e:
            print(f"Error in adding document: {str(e)}")
            raise


    # get all documents
    def get_documents(self) -> List[Dict]:
        try:
            docs = self.document_processor.get_all_documents()
            return docs
        except Exception as e:
            print(f"Error in getting documents: {str(e)}")
            return []


    # delete document
    def delete_document(self, doc_id: str) -> bool:
        try:
            success = self.document_processor.delete_document(doc_id)
            return success
        except Exception as e:
            print(f"Error in deleting document: {str(e)}")
            return False


    # total count of documents
    def get_document_count(self) -> int:
        try:
            count = self.document_processor.get_document_count()
            return count
        except Exception as e:
            print(f"Error getting document count: {str(e)}")
            return 0
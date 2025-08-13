"""
ChromaDB Operations Module
Handles all ChromaDB-related operations including loading, querying, and reranking
"""

import os
from typing import List, Dict, Any, Optional
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from sentence_transformers import CrossEncoder
from groq import Groq
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class ChromaDBManager:
    """Manager class for ChromaDB operations"""
    
    def __init__(
        self,
        db_path: str = "./chroma_db_hf",
        collection_name: Optional[str] = None,
        embedding_model_name: str = "BAAI/bge-small-en-v1.5",
        reranker_model_name: str = "BAAI/bge-reranker-base"
    ):
        """
        Initialize ChromaDB Manager
        
        Args:
            db_path: Path to ChromaDB directory
            collection_name: Optional name of the collection. If None, use Chroma's default.
            embedding_model_name: Name of the embedding model
            reranker_model_name: Name of the reranker model
        """
        self.db_path = db_path
        self.collection_name = collection_name
        
        # Initialize embedding model
        self.embedding_model = HuggingFaceEmbeddings(
            model_name=embedding_model_name,
        )
        
        # Initialize reranker model
        self.reranker_model = CrossEncoder(reranker_model_name)
        
        # Initialize Groq client if API key is available
        self.groq_client = None
        if os.getenv("GROQ_API_KEY"):
            self.groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        
        # Load ChromaDB
        self.chroma_db = self._load_chroma_db()
    
    def _load_chroma_db(self) -> Optional[Chroma]:
        """Load existing ChromaDB instance"""
        try:
            if not os.path.exists(self.db_path):
                print(f"ChromaDB path {self.db_path} does not exist")
                return None
                
            # Only pass collection_name if explicitly provided; otherwise use default
            if self.collection_name:
                chroma_db = Chroma(
                    collection_name=self.collection_name,
                    embedding_function=self.embedding_model,
                    persist_directory=self.db_path,
                )
            else:
                chroma_db = Chroma(
                    embedding_function=self.embedding_model,
                    persist_directory=self.db_path,
                )
            
            # Check if collection has documents
            collection = chroma_db._collection
            if collection.count() == 0:
                coll = self.collection_name if self.collection_name else "<default>"
                print(f"ChromaDB collection {coll} is empty")
                return None
                
            return chroma_db
        except Exception as e:
            print(f"Error loading ChromaDB: {e}")
            return None
    
    def query_similarity(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """
        Perform similarity search on ChromaDB
        
        Args:
            query: Search query
            top_k: Number of results to retrieve
            
        Returns:
            List of chunks with metadata
        """
        if not self.chroma_db:
            return []
        
        try:
            results = self.chroma_db.similarity_search(
                query=query,
                k=top_k
            )
            
            chunks = []
            for doc in results:
                chunks.append({
                    'page_content': doc.page_content,
                    'metadata': doc.metadata
                })
            
            return chunks
        except Exception as e:
            print(f"Error in similarity search: {e}")
            return []
    
    def query_with_scores(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """
        Perform similarity search with scores
        
        Args:
            query: Search query
            top_k: Number of results to retrieve
            
        Returns:
            List of chunks with metadata and similarity scores
        """
        if not self.chroma_db:
            return []
        
        try:
            results = self.chroma_db.similarity_search_with_score(
                query=query,
                k=top_k
            )
            
            chunks = []
            for doc, score in results:
                chunks.append({
                    'page_content': doc.page_content,
                    'metadata': doc.metadata,
                    'similarity_score': float(score)
                })
            
            return chunks
        except Exception as e:
            print(f"Error in similarity search with scores: {e}")
            return []
    
    def rerank_chunks(self, query: str, chunks: List[Dict], top_k: int = 5) -> List[Dict]:
        """
        Rerank chunks using cross-encoder model
        
        Args:
            query: Original query
            chunks: List of chunks to rerank
            top_k: Number of top chunks to return
            
        Returns:
            Reranked chunks with scores
        """
        if not chunks:
            return []
        
        try:
            # Prepare pairs for reranking
            pairs = [[query, chunk['page_content']] for chunk in chunks]
            
            # Get scores from reranker
            scores = self.reranker_model.predict(pairs)
            
            # Add scores to chunks
            for i, chunk in enumerate(chunks):
                chunk['rerank_score'] = float(scores[i])
            
            # Sort by rerank score (descending)
            ranked_chunks = sorted(chunks, key=lambda x: x['rerank_score'], reverse=True)
            
            return ranked_chunks[:top_k]
        except Exception as e:
            print(f"Error in reranking: {e}")
            return chunks[:top_k]
    
    def search_by_file(self, file_path: str, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Search for chunks from a specific file
        
        Args:
            file_path: Path of the file to search for
            limit: Maximum number of chunks to return
            
        Returns:
            List of chunks from the specified file
        """
        if not self.chroma_db:
            return []
        
        try:
            # Query with metadata filter
            results = self.chroma_db.get(
                where={"file_path": {"$eq": file_path}},
                limit=limit
            )
            
            chunks = []
            if results and 'documents' in results:
                for i, doc in enumerate(results['documents']):
                    chunk = {
                        'page_content': doc,
                        'metadata': results['metadatas'][i] if 'metadatas' in results else {}
                    }
                    chunks.append(chunk)
            
            return chunks
        except Exception as e:
            print(f"Error searching by file: {e}")
            return []
    
    def get_database_info(self) -> Dict[str, Any]:
        """
        Get information about the ChromaDB database
        
        Returns:
            Dictionary with database statistics
        """
        if not self.chroma_db:
            return {
                "exists": False,
                "error": "ChromaDB not loaded"
            }
        
        try:
            collection = self.chroma_db._collection
            
            return {
                "exists": True,
                "database_path": self.db_path,
                "collection_name": self.collection_name,
                "document_count": collection.count(),
                "embedding_model": "BAAI/bge-small-en-v1.5",
                "reranker_model": "BAAI/bge-reranker-base"
            }
        except Exception as e:
            return {
                "exists": False,
                "error": f"Error getting database info: {str(e)}"
            }
    
    def build_context(self, chunks: List[Dict]) -> str:
        """
        Build context from chunks for LLM
        
        Args:
            chunks: List of chunks
            
        Returns:
            Formatted context string
        """
        context = "## Relevant Code Context:\n\n"
        for i, chunk in enumerate(chunks, 1):
            context += f"### Chunk {i}:\n"
            context += f"**File**: {chunk.get('metadata', {}).get('file_path', 'Unknown')}\n"
            
            # Add score if available
            if 'rerank_score' in chunk:
                context += f"**Rerank Score**: {chunk['rerank_score']:.4f}\n"
            elif 'similarity_score' in chunk:
                context += f"**Similarity Score**: {chunk['similarity_score']:.4f}\n"
            
            context += f"```python\n{chunk['page_content']}\n```\n\n"
        
        return context
    
    def generate_llm_response(self, query: str, chunks: List[Dict]) -> Optional[str]:
        """
        Generate LLM response based on retrieved chunks
        
        Args:
            query: User query
            chunks: Retrieved chunks
            
        Returns:
            LLM response or None if not available
        """
        if not self.groq_client or not chunks:
            return None
        
        try:
            context = self.build_context(chunks)
            
            messages = [
                {
                    "role": "system",
                    "content": "You are a helpful assistant that answers questions based on the provided code context. Be concise and specific."
                },
                {
                    "role": "user",
                    "content": f"Based on the following context, answer this question: {query}\n\n{context}"
                }
            ]
            
            completion = self.groq_client.chat.completions.create(
                model="llama-3.1-70b-versatile",
                messages=messages,
                temperature=0.3,
                max_tokens=1000
            )
            
            return completion.choices[0].message.content
        except Exception as e:
            return f"Error generating LLM response: {str(e)}"

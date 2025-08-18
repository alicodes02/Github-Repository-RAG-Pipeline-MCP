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
from .logging_config import get_logger

# Load environment variables
load_dotenv()

# Initialize logger (will use existing configuration)
logger = get_logger(__name__)

class ChromaDBManager:
    """Manager class for ChromaDB operations"""
    
    def __init__(
        self,
        db_path: str = None,
        collection_name: Optional[str] = None,
        embedding_model_name: str = "BAAI/bge-small-en-v1.5",
        reranker_model_name: str = "BAAI/bge-reranker-base"
    ):
        """
        Initialize ChromaDB Manager
        
        Args:
            db_path: Path to ChromaDB directory. If None, uses DB_PATH environment variable or default
            collection_name: Optional name of the collection. If None, use Chroma's default.
            embedding_model_name: Name of the embedding model
            reranker_model_name: Name of the reranker model
        """
        logger.info("Initializing ChromaDBManager")
        
        # Handle DB path with proper fallback - ensure absolute path to root directory
        if db_path is None:
            # Get the root directory of the project (parent of rag_mcp)
            current_dir = os.path.dirname(os.path.abspath(__file__))  # rag_mcp directory
            project_root = os.path.dirname(current_dir)  # parent directory (project root)
            default_db_path = os.path.join(project_root, "chroma_db_hf")
            db_path = os.getenv("DB_PATH", default_db_path)
        
        # Ensure the path is absolute
        if not os.path.isabs(db_path):
            current_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.dirname(current_dir)
            db_path = os.path.join(project_root, db_path)
        
        self.db_path = os.path.abspath(db_path)
        logger.debug(f"DB path (absolute): {self.db_path}")
        logger.debug(f"Collection name: {collection_name}")
        logger.debug(f"Embedding model: {embedding_model_name}")
        logger.debug(f"Reranker model: {reranker_model_name}")
        
        self.collection_name = collection_name
        
        try:
            # Initialize embedding model
            logger.info(f"Loading embedding model: {embedding_model_name}")
            self.embedding_model = HuggingFaceEmbeddings(
                model_name=embedding_model_name,
            )
            logger.info("Embedding model loaded successfully")
            
            # Initialize reranker model
            logger.info(f"Loading reranker model: {reranker_model_name}")
            self.reranker_model = CrossEncoder(reranker_model_name)
            logger.info("Reranker model loaded successfully")
            
            # Initialize Groq client if API key is available
            self.groq_client = None
            if os.getenv("GROQ_API_KEY"):
                logger.info("Initializing Groq client")
                self.groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
                logger.info("Groq client initialized successfully")
            else:
                logger.warning("GROQ_API_KEY not found - LLM responses will be unavailable")
            
            # Load ChromaDB
            logger.info("Loading ChromaDB instance")
            self.chroma_db = self._load_chroma_db()
            
            if self.chroma_db:
                logger.info("ChromaDBManager initialization completed successfully")
            else:
                logger.warning("ChromaDBManager initialized but ChromaDB is not available")
                
        except Exception as e:
            logger.error(f"Error during ChromaDBManager initialization: {e}", exc_info=True)
            raise
    
    def _load_chroma_db(self) -> Optional[Chroma]:
        """Load existing ChromaDB instance"""
        logger.debug(f"Attempting to load ChromaDB from: {self.db_path}")
        
        try:
            if not os.path.exists(self.db_path):
                logger.warning(f"ChromaDB path {self.db_path} does not exist")
                return None
            
            logger.debug("ChromaDB path exists, loading database")
            
            # Only pass collection_name if explicitly provided; otherwise use default
            if self.collection_name:
                logger.debug(f"Using specific collection: {self.collection_name}")
                chroma_db = Chroma(
                    collection_name=self.collection_name,
                    embedding_function=self.embedding_model,
                    persist_directory=self.db_path,
                )
            else:
                logger.debug("Using default collection")
                chroma_db = Chroma(
                    embedding_function=self.embedding_model,
                    persist_directory=self.db_path,
                )
            
            # Check if collection has documents
            collection = chroma_db._collection
            doc_count = collection.count()
            logger.info(f"ChromaDB loaded with {doc_count} documents")
            
            if doc_count == 0:
                coll = self.collection_name if self.collection_name else "<default>"
                logger.warning(f"ChromaDB collection {coll} is empty")
                return None
                
            return chroma_db
        except Exception as e:
            logger.error(f"Error loading ChromaDB: {e}", exc_info=True)
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
        logger.info(f"Performing similarity search for query: '{query}' (top_k={top_k})")
        
        if not self.chroma_db:
            logger.error("ChromaDB not available for similarity search")
            return []
        
        try:
            logger.debug("Executing similarity search")
            results = self.chroma_db.similarity_search(
                query=query,
                k=top_k
            )
            
            logger.info(f"Similarity search returned {len(results)} results")
            
            chunks = []
            for i, doc in enumerate(results):
                logger.debug(f"Processing result {i+1}: {doc.metadata.get('file_path', 'Unknown')}")
                chunks.append({
                    'page_content': doc.page_content,
                    'metadata': doc.metadata
                })
            
            logger.info(f"Successfully processed {len(chunks)} chunks")
            return chunks
        except Exception as e:
            logger.error(f"Error in similarity search: {e}", exc_info=True)
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
        logger.info(f"Performing similarity search with scores for query: '{query}' (top_k={top_k})")
        
        if not self.chroma_db:
            logger.error("ChromaDB not available for similarity search with scores")
            return []
        
        try:
            logger.debug("Executing similarity search with scores")
            results = self.chroma_db.similarity_search_with_score(
                query=query,
                k=top_k
            )
            
            logger.info(f"Similarity search with scores returned {len(results)} results")
            
            chunks = []
            for i, (doc, score) in enumerate(results):
                logger.debug(f"Processing result {i+1}: score={score:.4f}, file={doc.metadata.get('file_path', 'Unknown')}")
                chunks.append({
                    'page_content': doc.page_content,
                    'metadata': doc.metadata,
                    'similarity_score': float(score)
                })
            
            logger.info(f"Successfully processed {len(chunks)} chunks with scores")
            return chunks
        except Exception as e:
            logger.error(f"Error in similarity search with scores: {e}", exc_info=True)
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
        logger.info(f"Reranking {len(chunks)} chunks for query: '{query}' (top_k={top_k})")
        
        if not chunks:
            logger.warning("No chunks provided for reranking")
            return []
        
        try:
            logger.debug("Preparing pairs for reranking")
            # Prepare pairs for reranking
            pairs = [[query, chunk['page_content']] for chunk in chunks]
            
            logger.debug(f"Running reranker model on {len(pairs)} pairs")
            # Get scores from reranker
            scores = self.reranker_model.predict(pairs)
            
            logger.debug("Adding rerank scores to chunks")
            # Add scores to chunks
            for i, chunk in enumerate(chunks):
                chunk['rerank_score'] = float(scores[i])
                logger.debug(f"Chunk {i+1} rerank score: {scores[i]:.4f}")
            
            # Sort by rerank score (descending)
            logger.debug("Sorting chunks by rerank score")
            ranked_chunks = sorted(chunks, key=lambda x: x['rerank_score'], reverse=True)
            
            result = ranked_chunks[:top_k]
            logger.info(f"Reranking completed. Returning top {len(result)} chunks")
            
            return result
        except Exception as e:
            logger.error(f"Error in reranking: {e}", exc_info=True)
            logger.warning(f"Returning original chunks without reranking")
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
        logger.info(f"Searching for chunks from file: '{file_path}' (limit={limit})")
        
        if not self.chroma_db:
            logger.error("ChromaDB not available for file search")
            return []
        
        try:
            logger.debug(f"Executing file search with metadata filter")
            # Query with metadata filter
            results = self.chroma_db.get(
                where={"file_path": {"$eq": file_path}},
                limit=limit
            )
            
            chunks = []
            if results and 'documents' in results:
                logger.info(f"Found {len(results['documents'])} chunks for file: {file_path}")
                for i, doc in enumerate(results['documents']):
                    chunk = {
                        'page_content': doc,
                        'metadata': results['metadatas'][i] if 'metadatas' in results else {}
                    }
                    chunks.append(chunk)
                    logger.debug(f"Processed chunk {i+1} from file search")
            else:
                logger.warning(f"No chunks found for file: {file_path}")
            
            logger.info(f"File search completed. Returning {len(chunks)} chunks")
            return chunks
        except Exception as e:
            logger.error(f"Error searching by file: {e}", exc_info=True)
            return []
    
    def get_database_info(self) -> Dict[str, Any]:
        """
        Get information about the ChromaDB database
        
        Returns:
            Dictionary with database statistics
        """
        logger.info("Retrieving database information")
        
        if not self.chroma_db:
            logger.warning("ChromaDB not available for info retrieval")
            return {
                "exists": False,
                "error": "ChromaDB not loaded"
            }
        
        try:
            collection = self.chroma_db._collection
            doc_count = collection.count()
            
            info = {
                "exists": True,
                "database_path": self.db_path,
                "collection_name": self.collection_name,
                "document_count": doc_count,
                "embedding_model": "BAAI/bge-small-en-v1.5",
                "reranker_model": "BAAI/bge-reranker-base"
            }
            
            logger.info(f"Database info retrieved: {doc_count} documents")
            logger.debug(f"Full database info: {info}")
            
            return info
        except Exception as e:
            logger.error(f"Error getting database info: {e}", exc_info=True)
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
        logger.info(f"Building context from {len(chunks)} chunks")
        
        context = "## Relevant Code Context:\n\n"
        for i, chunk in enumerate(chunks, 1):
            logger.debug(f"Processing chunk {i} for context building")
            context += f"### Chunk {i}:\n"
            context += f"**File**: {chunk.get('metadata', {}).get('file_path', 'Unknown')}\n"
            
            # Add score if available
            if 'rerank_score' in chunk:
                context += f"**Rerank Score**: {chunk['rerank_score']:.4f}\n"
            elif 'similarity_score' in chunk:
                context += f"**Similarity Score**: {chunk['similarity_score']:.4f}\n"
            
            context += f"```python\n{chunk['page_content']}\n```\n\n"
        
        logger.info("Context building completed")
        logger.debug(f"Context length: {len(context)} characters")
        
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
        logger.info(f"Generating LLM response for query: '{query}' with {len(chunks)} chunks")
        
        if not self.groq_client:
            logger.warning("Groq client not available - cannot generate LLM response")
            return None
            
        if not chunks:
            logger.warning("No chunks provided for LLM response generation")
            return None
        
        try:
            logger.debug("Building context for LLM")
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
            
            logger.debug("Sending request to Groq API")
            completion = self.groq_client.chat.completions.create(
                model="llama-3.1-70b-versatile",
                messages=messages,
                temperature=0.3,
                max_tokens=1000
            )
            
            response = completion.choices[0].message.content
            logger.info("LLM response generated successfully")
            logger.debug(f"Response length: {len(response)} characters")
            
            return response
        except Exception as e:
            logger.error(f"Error generating LLM response: {e}", exc_info=True)
            return f"Error generating LLM response: {str(e)}"

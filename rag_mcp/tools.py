"""
MCP Tools Module
Defines all tools exposed by the MCP server
"""

from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
from .chromadb_operations import ChromaDBManager
import sys
import os

# Initialize ChromaDB Manager
db_manager = ChromaDBManager()

# Import the function from main.py
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from main import load_github_repository, convert_chunks_to_langchain_docs, load_or_create_vector_store

# Request/Response Models
class QueryRequest(BaseModel):
    """Request model for ChromaDB query with reranking"""
    query: str = Field(..., description="The query to search for in the ChromaDB")
    top_k: int = Field(default=10, description="Number of top results to retrieve before reranking")
    rerank_top_k: int = Field(default=5, description="Number of top results to return after reranking")
    include_llm_response: bool = Field(default=False, description="Whether to include an LLM-generated response based on the retrieved context")

class ChunksRequest(BaseModel):
    """Request model for getting raw chunks without reranking"""
    query: str = Field(..., description="The query to search for in the ChromaDB")
    top_k: int = Field(default=10, description="Number of chunks to retrieve")
    include_scores: bool = Field(default=True, description="Whether to include similarity scores")

class QueryResponse(BaseModel):
    """Response model for ChromaDB query"""
    success: bool
    query: str
    results: List[Dict[str, Any]]
    llm_response: Optional[str] = None
    error: Optional[str] = None

class ChunksResponse(BaseModel):
    """Response model for raw chunks"""
    success: bool
    query: str
    chunks: List[Dict[str, Any]]
    total_chunks: int
    error: Optional[str] = None

class LoadGithubRepoRequest(BaseModel):
    """Request model for loading a GitHub repository and chunking its files"""
    url: str = Field(..., description="GitHub repository URL")
    branch: str = Field(default="master", description="Branch name to load")

class LoadGithubRepoResponse(BaseModel):
    """Response model for loaded chunks from GitHub repository"""
    success: bool
    repo_url: str
    branch: str
    total_chunks: int
    chunks: List[Dict[str, Any]]
    chromadb_result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None

def query_chromadb(request: QueryRequest) -> QueryResponse:
    """
    Query the ChromaDB vector database with semantic search and reranking.
    
    This tool:
    1. Searches ChromaDB for relevant documents using semantic similarity
    2. Reranks the results using a cross-encoder model
    3. Optionally generates an LLM response based on the retrieved context
    
    Args:
        request: QueryRequest containing the query and parameters
        
    Returns:
        QueryResponse with the search results and optional LLM response
    """
    try:
        if not db_manager.chroma_db:
            return QueryResponse(
                success=False,
                query=request.query,
                results=[],
                error="ChromaDB not found or empty. Please ensure the database exists at ./chroma_db_hf"
            )
        
        # Perform similarity search
        chunks = db_manager.query_similarity(request.query, request.top_k)
        
        if not chunks:
            return QueryResponse(
                success=True,
                query=request.query,
                results=[],
                error="No results found for the query"
            )
        
        # Rerank the chunks
        ranked_chunks = db_manager.rerank_chunks(request.query, chunks, request.rerank_top_k)
        
        # Prepare response
        response = QueryResponse(
            success=True,
            query=request.query,
            results=ranked_chunks
        )
        
        # Generate LLM response if requested
        if request.include_llm_response:
            llm_response = db_manager.generate_llm_response(request.query, ranked_chunks)
            response.llm_response = llm_response
        
        return response
        
    except Exception as e:
        return QueryResponse(
            success=False,
            query=request.query,
            results=[],
            error=f"Error querying ChromaDB: {str(e)}"
        )

def get_chunks(request: ChunksRequest) -> ChunksResponse:
    """
    Get raw chunks from ChromaDB without reranking.
    
    This tool retrieves chunks based on semantic similarity without applying
    any reranking. Useful for getting more results or when reranking is not needed.
    
    Args:
        request: ChunksRequest containing the query and parameters
        
    Returns:
        ChunksResponse with the raw chunks
    """
    try:
        if not db_manager.chroma_db:
            return ChunksResponse(
                success=False,
                query=request.query,
                chunks=[],
                total_chunks=0,
                error="ChromaDB not found or empty. Please ensure the database exists at ./chroma_db_hf"
            )
        
        # Get chunks with or without scores
        if request.include_scores:
            chunks = db_manager.query_with_scores(request.query, request.top_k)
        else:
            chunks = db_manager.query_similarity(request.query, request.top_k)
        
        return ChunksResponse(
            success=True,
            query=request.query,
            chunks=chunks,
            total_chunks=len(chunks)
        )
        
    except Exception as e:
        return ChunksResponse(
            success=False,
            query=request.query,
            chunks=[],
            total_chunks=0,
            error=f"Error getting chunks: {str(e)}"
        )

def get_database_info() -> Dict[str, Any]:
    """
    Get information about the ChromaDB database.
    
    Returns:
        Dictionary containing database statistics and metadata
    """
    try:
        info = db_manager.get_database_info()
        info["success"] = info.get("exists", False)
        return info
    except Exception as e:
        return {
            "success": False,
            "error": f"Error getting database info: {str(e)}"
        }

def search_by_file(file_path: str, limit: int = 10) -> Dict[str, Any]:
    """
    Search for chunks from a specific file in the ChromaDB.
    
    Args:
        file_path: Path of the file to search for
        limit: Maximum number of chunks to return
        
    Returns:
        Dictionary containing the search results
    """
    try:
        if not db_manager.chroma_db:
            return {
                "success": False,
                "error": "ChromaDB not found or empty"
            }
        
        chunks = db_manager.search_by_file(file_path, limit)
        
        return {
            "success": True,
            "file_path": file_path,
            "chunk_count": len(chunks),
            "chunks": chunks
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": f"Error searching by file: {str(e)}"
        }

def load_github_repository_tool(request: LoadGithubRepoRequest) -> LoadGithubRepoResponse:
    """
    Tool to load a GitHub repository, chunk its files, convert to LangChain docs, and insert into ChromaDB.
    
    This tool:
    1. Loads and chunks a GitHub repository
    2. Converts chunks to LangChain Document format
    3. Adds the documents to ChromaDB vector store
    4. Returns chunk metadata and ChromaDB insertion results
    
    Args:
        request: LoadGithubRepoRequest containing repo URL and branch
    Returns:
        LoadGithubRepoResponse with chunk metadata and ChromaDB results
    """
    try:
        # Step 1: Load and chunk GitHub repository
        print(f"Loading GitHub repository: {request.url}")
        chunks = load_github_repository(url=request.url, branch=request.branch)
        
        if not chunks:
            return LoadGithubRepoResponse(
                success=False,
                repo_url=request.url,
                branch=request.branch,
                total_chunks=0,
                chunks=[],
                error="No chunks were generated from the repository"
            )
        
        # Step 2: Convert chunks to LangChain documents
        print(f"Converting {len(chunks)} chunks to LangChain documents...")
        langchain_docs = convert_chunks_to_langchain_docs(chunks)
        
        # Step 3: Add documents to ChromaDB using load_or_create_vector_store
        print("Adding documents to ChromaDB vector store...")
        chroma_db = load_or_create_vector_store(langchain_docs=langchain_docs, force_recreate=False)
        
        # Get ChromaDB statistics
        doc_count = chroma_db._collection.count() if chroma_db._collection else 0
        
        chromadb_result = {
            "success": True,
            "total_documents_in_db": doc_count,
            "documents_added": len(langchain_docs),
            "message": f"Successfully added {len(langchain_docs)} documents to ChromaDB"
        }
        
        return LoadGithubRepoResponse(
            success=True,
            repo_url=request.url,
            branch=request.branch,
            total_chunks=len(chunks),
            chunks=[{
                "text": chunk["text"],
                "metadata": chunk["metadata"]
            } for chunk in chunks],
            chromadb_result=chromadb_result
        )
        
    except Exception as e:
        return LoadGithubRepoResponse(
            success=False,
            repo_url=request.url,
            branch=request.branch,
            total_chunks=0,
            chunks=[],
            chromadb_result={
                "success": False,
                "error": f"ChromaDB insertion failed: {str(e)}"
            },
            error=f"Error loading GitHub repository: {str(e)}"
        )

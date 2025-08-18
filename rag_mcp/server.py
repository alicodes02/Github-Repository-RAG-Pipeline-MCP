#!/usr/bin/env python3
"""
MCP Server for ChromaDB RAG Pipeline
This server provides tools to query a ChromaDB vector database
"""

import sys
import os
# Add parent directory to path to import from root
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastmcp import FastMCP
from rag_mcp.tools import (
    query_chromadb,
    get_chunks,
    get_database_info,
    search_by_file,
    load_github_repository_tool,
    QueryRequest,
    ChunksRequest,
    LoadGithubRepoRequest
)

# Initialize FastMCP server
mcp = FastMCP("ChromaDB RAG Server")

# Register tools with the MCP server
@mcp.tool()
def query_chromadb_tool(
    
    query: str,
    top_k: int = 10,
    rerank_top_k: int = 5,
    include_llm_response: bool = False
    
    ):
    
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

    request = QueryRequest(
        query=query,
        top_k=top_k,
        rerank_top_k=rerank_top_k,
        include_llm_response=include_llm_response
    )

    return query_chromadb(request)

@mcp.tool()
def get_chunks_tool(request: ChunksRequest):
    """
    Get raw chunks from ChromaDB without reranking.
    
    This tool retrieves chunks based on semantic similarity without applying
    any reranking. Useful for getting more results or when reranking is not needed.
    
    Args:
        request: ChunksRequest containing the query and parameters
        
    Returns:
        ChunksResponse with the raw chunks
    """
    return get_chunks(request)

@mcp.tool()
def get_database_info_tool():
    """
    Get information about the ChromaDB database.
    
    Returns:
        Dictionary containing database statistics and metadata
    """
    return get_database_info()

@mcp.tool()
def search_by_file_tool(file_path: str, limit: int = 10):
    """
    Search for chunks from a specific file in the ChromaDB.
    
    Args:
        file_path: Path of the file to search for
        limit: Maximum number of chunks to return
        
    Returns:
        Dictionary containing the search results
    """
    return search_by_file(file_path, limit)

@mcp.tool()
def load_github_repo(url: str, branch: str = "main"):
    """
    Load a GitHub repository into the ChromaDB vector database.
    
    Args:
        url: URL of the GitHub repository to load
        branch: Branch name to load (defaults to "master")
        
    Returns:
        LoadGithubRepoResponse with chunk metadata
    """
    # from rag_mcp.tools import LoadGithubRepoRequest
    request = LoadGithubRepoRequest(url=url, branch=branch)
    return load_github_repository_tool(request)

if __name__ == "__main__":
    # Run the MCP server
    mcp.run()

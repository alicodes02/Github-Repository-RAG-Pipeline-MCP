#!/usr/bin/env python3
"""
MCP Server for ChromaDB RAG Pipeline
This server provides tools to query a ChromaDB vector database
"""

import sys
import os
import argparse
# Add parent directory to path to import from root
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastmcp import FastMCP
from rag_mcp.logging_config import get_logger
from rag_mcp.tools import (
    query_chromadb,
    get_database_info,
    search_by_file,
    load_github_repository_tool,
    QueryRequest,
    LoadGithubRepoRequest
)

# Initialize logger (will use existing configuration)
logger = get_logger(__name__)

logger.info("Starting MCP Server for ChromaDB RAG Pipeline")

# Initialize FastMCP server
mcp = FastMCP("ChromaDB RAG Server")
logger.info("FastMCP server initialized")

# Register tools with the MCP server
@mcp.tool()
def query_chromadb_tool(
    query: str,
    top_k: int = 10,
    rerank_top_k: int = 5,
):
    """
    Query the ChromaDB vector database with semantic search and reranking.
    
    This tool:
    1. Searches ChromaDB for relevant documents using semantic similarity
    2. Reranks the results using a cross-encoder model
    3. Optionally generates an LLM response based on the retrieved context
    
    Args:
        query: The query to search for in the ChromaDB
        top_k: Number of top results to retrieve before reranking (default: 10)
        rerank_top_k: Number of top results to return after reranking (default: 5)
        
    Returns:
        QueryResponse with the search results and optional LLM response
    """

    logger.info(f"MCP Tool called: query_chromadb_tool - Query: '{query}'")
    logger.debug(f"Parameters: top_k={top_k}, rerank_top_k={rerank_top_k}")

    try:
        request = QueryRequest(
            query=query,
            top_k=top_k,
            rerank_top_k=rerank_top_k,
        )

        logger.debug("QueryRequest created, calling query_chromadb")
        result = query_chromadb(request)
        logger.info(f"query_chromadb_tool completed: success={result.success}")
        return result
    except Exception as e:
        logger.error(f"Error in query_chromadb_tool: {e}", exc_info=True)
        raise


@mcp.tool()
def get_database_info_tool():
    """
    Get information about the ChromaDB database.
    
    Returns:
        Dictionary containing database statistics and metadata
    """
    logger.info("MCP Tool called: get_database_info_tool")
    
    try:
        result = get_database_info()
        logger.info(f"get_database_info_tool completed: success={result.get('success', False)}")
        return result
    except Exception as e:
        logger.error(f"Error in get_database_info_tool: {e}", exc_info=True)
        raise

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
    logger.info(f"MCP Tool called: search_by_file_tool - File: '{file_path}', limit: {limit}")
    
    try:
        result = search_by_file(file_path, limit)
        logger.info(f"search_by_file_tool completed: success={result.get('success', False)}, chunks={result.get('chunk_count', 0)}")
        return result
    except Exception as e:
        logger.error(f"Error in search_by_file_tool: {e}", exc_info=True)
        raise

@mcp.tool()
def load_github_repo(url: str, branch: str = "main"):
    """
    Load a GitHub repository into the ChromaDB vector database.
    
    Args:
        url: URL of the GitHub repository to load
        branch: Branch name to load (defaults to "main")
        
    Returns:
        LoadGithubRepoResponse with chunk metadata
    """
    logger.info(f"MCP Tool called: load_github_repo - URL: '{url}', Branch: '{branch}'")
    
    try:
        request = LoadGithubRepoRequest(url=url, branch=branch)
        logger.debug("LoadGithubRepoRequest created, calling load_github_repository_tool")
        
        result = load_github_repository_tool(request)
        logger.info(f"load_github_repo completed: success={result.success}, chunks={result.total_chunks}")
        return result
    except Exception as e:
        logger.error(f"Error in load_github_repo: {e}", exc_info=True)
        raise

if __name__ == "__main__":
    logger.info("Starting MCP server main execution")
    # Parse optional CLI for GitHub token (tolerate unknown args due to fastmcp)
    try:
        parser = argparse.ArgumentParser(add_help=False)
        parser.add_argument("--github-token", dest="github_token", default=None)
        # Ignore unknown args so FastMCP's own args don't break us
        args, _unknown = parser.parse_known_args(sys.argv[1:])
        if args.github_token:
            os.environ["GITHUB_TOKEN"] = args.github_token
            logger.info("GITHUB_TOKEN set from CLI argument")
        else:
            logger.info("No --github-token provided; will use environment if present")
    except Exception as e:
        logger.warning(f"Failed to parse CLI args for github token: {e}")
    # Preload GitHub repo into ChromaDB on startup
    try:
        repo_url = "https://github.com/Genie-Experiments/rag-vs-llamaparse"
        logger.info(f"Preloading GitHub repository at startup: {repo_url}")
        preload_request = LoadGithubRepoRequest(url=repo_url, branch="main")
        preload_result = load_github_repository_tool(preload_request)
        logger.info(f"Preload completed: success={preload_result.success}, chunks={preload_result.total_chunks}")
    except Exception as e:
        logger.error(f"Failed to preload GitHub repository at startup: {e}", exc_info=True)
    try:
        # Run the MCP server
        logger.info("Running MCP server...")
        mcp.run(transport="sse", host="0.0.0.0", port=8010)
    except KeyboardInterrupt:
        logger.info("MCP server stopped by user (KeyboardInterrupt)")
    except Exception as e:
        logger.error(f"MCP server error: {e}", exc_info=True)
        raise
    finally:
        logger.info("MCP server shutdown")
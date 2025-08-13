# ChromaDB RAG MCP Server

This MCP (Model Context Protocol) server provides tools to query your ChromaDB vector database using FastMCP 2.0.

## Architecture

The MCP server is organized into modular components:

- **`chromadb_operations.py`** - Core ChromaDB operations and database management
- **`tools.py`** - Tool definitions and request/response models
- **`server.py`** - Main MCP server that registers and exposes the tools

## Features

The server exposes four main tools:

1. **`query_chromadb_tool`** - Main query tool with semantic search and reranking
2. **`get_chunks_tool`** - Get raw chunks without reranking (NEW)
3. **`get_database_info_tool`** - Get information about the ChromaDB database
4. **`search_by_file_tool`** - Search for chunks from a specific file

## Installation

1. Install the required dependencies from the root directory:
```bash
pip install -r requirements.txt
```

2. Ensure you have a `.env` file in the root directory with the following variables:
```
GROQ_API_KEY=your_groq_api_key_here  # Required for LLM responses
GITHUB_TOKEN=your_github_token_here  # If using GitHub repository loading
```

## Usage

### Running the MCP Server

From the root directory, run:
```bash
python mcp/server.py
```

Or use it with an MCP client that supports the protocol using the `mcp_config.json`.

### Available Tools

#### 1. query_chromadb_tool

Query the ChromaDB with semantic search and reranking.

**Parameters:**
- `query` (str, required): The search query
- `top_k` (int, default=10): Number of results to retrieve before reranking
- `rerank_top_k` (int, default=5): Number of results after reranking
- `include_llm_response` (bool, default=False): Include LLM-generated response

**Example Request:**
```json
{
  "query": "How does the authentication system work?",
  "top_k": 10,
  "rerank_top_k": 5,
  "include_llm_response": true
}
```

**Response:**
```json
{
  "success": true,
  "query": "How does the authentication system work?",
  "results": [
    {
      "page_content": "...",
      "metadata": {
        "file_path": "auth.py"
      },
      "rerank_score": 0.95
    }
  ],
  "llm_response": "Based on the code context..."
}
```

#### 2. get_chunks_tool (NEW)

Get raw chunks from ChromaDB without reranking.

**Parameters:**
- `query` (str, required): The search query
- `top_k` (int, default=10): Number of chunks to retrieve
- `include_scores` (bool, default=True): Whether to include similarity scores

**Example Request:**
```json
{
  "query": "database connection",
  "top_k": 20,
  "include_scores": true
}
```

**Response:**
```json
{
  "success": true,
  "query": "database connection",
  "chunks": [
    {
      "page_content": "...",
      "metadata": {
        "file_path": "db.py"
      },
      "similarity_score": 0.89
    }
  ],
  "total_chunks": 20
}
```

#### 3. get_database_info_tool

Get information about the ChromaDB database.

**Response:**
```json
{
  "success": true,
  "database_path": "./chroma_db_hf",
  "collection_name": "github_docs",
  "document_count": 150,
  "embedding_model": "BAAI/bge-small-en-v1.5",
  "reranker_model": "BAAI/bge-reranker-base"
}
```

#### 4. search_by_file_tool

Search for chunks from a specific file.

**Parameters:**
- `file_path` (str, required): Path of the file to search for
- `limit` (int, default=10): Maximum number of chunks to return

**Example Request:**
```json
{
  "file_path": "src/main.py",
  "limit": 5
}
```

## Testing the Server

You can test the server using a simple Python script:

```python
import sys
sys.path.append('..')  # Add parent directory to path

from mcp.tools import QueryRequest, ChunksRequest, query_chromadb, get_chunks

# Test query with reranking
request = QueryRequest(
    query="What does this repository do?",
    top_k=10,
    rerank_top_k=5,
    include_llm_response=True
)
response = query_chromadb(request)
print("Query with reranking:", response)

# Test getting raw chunks
chunks_request = ChunksRequest(
    query="database operations",
    top_k=15,
    include_scores=True
)
chunks_response = get_chunks(chunks_request)
print("Raw chunks:", chunks_response)
```

## File Structure

```
mcp/
├── chromadb_operations.py  # Core ChromaDB operations
├── tools.py                # Tool definitions and models
├── server.py               # Main MCP server
├── mcp_config.json        # MCP configuration
└── README.md              # This file
```

## Technologies Used

- **FastMCP 2.0** for the MCP protocol implementation
- **ChromaDB** for vector storage
- **HuggingFace BGE embeddings** for semantic search
- **BGE Reranker** for result reranking
- **Groq API** (optional) for LLM responses

## Prerequisites

Before using the MCP server, ensure you have:
1. A populated ChromaDB database at `./chroma_db_hf` (relative to project root)
2. The required Python packages installed
3. Environment variables configured (for LLM features)

## Troubleshooting

- **"ChromaDB not found"**: Ensure the database exists at `./chroma_db_hf` in the project root and has been populated with documents
- **"No GROQ_API_KEY"**: LLM responses require a Groq API key in your `.env` file
- **Import errors**: Run `pip install -r requirements.txt` from the root directory

## Integration with MCP Clients

This server can be used with any MCP-compatible client. Configure your client to connect to this server using the `mcp_config.json` file provided in this directory.

# GitHub Repository RAG Pipeline MCP

A FastMCP server that exposes tools to ingest GitHub repositories into a ChromaDB vector store and query them with semantic search and reranking.

## Tools (exposed by `rag_mcp/server.py`)

- __query_chromadb_tool__
  - Purpose: Semantic search + cross-encoder rerank; optional LLM answer.
  - Params:
    - `query: str`
    - `top_k: int = 10` (initial retrieval)
    - `rerank_top_k: int = 5` (final results)
    - `include_llm_response: bool = False`

- __get_chunks_tool__
  - Purpose: Raw semantic search results from ChromaDB without reranking.
  - Params:
    - `request: ChunksRequest` with `query: str`, `top_k: int = 10`, `include_scores: bool = True`

- __get_database_info_tool__
  - Purpose: Basic stats and metadata about the ChromaDB instance.
  - Params: none

- __search_by_file_tool__
  - Purpose: Return chunks for a specific `file_path` that were ingested.
  - Params:
    - `file_path: str`
    - `limit: int = 10`

- __load_github_repo__
  - Purpose: Load and chunk a GitHub repo, convert to LangChain docs, and insert into ChromaDB.
  - Params:
    - `url: str` (e.g., `https://github.com/owner/repo`)
    - `branch: str = "main"`
  - Notes: Uses `GITHUB_TOKEN` if set to increase API limits. Chunking uses semantic Python chunking for `.py` files and a recursive text splitter for others.

## Setup

1. __Create and activate a virtualenv__ (recommended)
   ```bash
   python -m venv venv
   source venv/bin/activate
   ```
2. __Install dependencies__
   ```bash
   pip install -r requirements.txt
   ```

## Environment variables

- __DB_PATH__ (optional): Path to the ChromaDB persist directory. Defaults to a project-root folder `./chroma_db_hf`.
- __GITHUB_TOKEN__ (optional but recommended): Ingestion via GitHub API. Can also be passed at runtime via `-- --github-token <TOKEN>`.
- __GROQ_API_KEY__ (optional): Enables LLM answer generation in `query_chromadb_tool` when `include_llm_response=True`.

You can place these in a `.env` file or export them in your shell before running the server.

## Run the MCP server

Run from the project root. Use the virtualenv Python to ensure correct packages are used.

```bash
./venv/bin/python -m fastmcp run rag_mcp/server.py --no-banner
```

To pass a GitHub token on the CLI (instead of environment):

```bash
./venv/bin/python -m fastmcp run rag_mcp/server.py --no-banner -- --github-token YOUR_TOKEN
```

The server name is "ChromaDB RAG Server" and it exposes the tools listed above.

## Notes

- Default embedding model: `BAAI/bge-small-en-v1.5`.
- Default reranker model: `BAAI/bge-reranker-base`.
- On first use, model weights may be downloaded.
- The ChromaDB path is resolved relative to the project root if a relative `DB_PATH` is provided.

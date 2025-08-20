"""
Ingestion utilities for loading GitHub repositories and converting to LangChain docs.
"""

import os
import re
import nest_asyncio
from typing import List, Dict, Any
from dotenv import load_dotenv
from llama_index.readers.github import GithubRepositoryReader, GithubClient
from langchain_text_splitters import (
    Language,
    RecursiveCharacterTextSplitter,
)
from langchain.schema import Document

# project-local chunker at repo root
from pyhton_chunker import PythonSemanticChunker

from .logging_config import get_logger

logger = get_logger(__name__)

# Ensure we can make nested asyncio calls when used inside other event loops
nest_asyncio.apply()
load_dotenv()

logger.info("Initializing ingestion module")


def parse_github_url(url: str):
    """Parse a GitHub URL into (owner, repo). Returns (None, None) if invalid."""
    logger.debug(f"Parsing GitHub URL: {url}")
    pattern = r"https://github\.com/([^/]+)/([^/]+)"
    match = re.match(pattern, url)
    result = match.groups() if match else (None, None)
    logger.debug(f"Parsed result: owner={result[0]}, repo={result[1]}")
    return result


def validate_owner_repo(owner: str, repo: str) -> bool:
    logger.debug(f"Validating owner: {owner}, repo: {repo}")
    is_valid = bool(owner) and bool(repo)
    logger.debug(f"Validation result: {is_valid}")
    return is_valid


def initialize_github_client(github_token: str | None = None) -> GithubClient:
    """Create a GithubClient using the provided token or env GITHUB_TOKEN."""
    token = github_token or os.getenv("GITHUB_TOKEN")
    if not token:
        logger.warning("GitHub token not provided and GITHUB_TOKEN env not set")
    else:
        logger.info("GitHub token provided for client creation")
    return GithubClient(token)


def load_github_repository(url: str, branch: str, github_token: str | None = None) -> List[Dict[str, Any]]:
    """Load a GitHub repository, chunk sources, and return raw chunks.

    Returns a list of dicts with keys {"text", "metadata"}.
    """
    logger.info(f"Starting GitHub repository loading - URL: {url}, Branch: {branch}")

    try:
        github_client = initialize_github_client(github_token)
        owner, repo = parse_github_url(url)

        if not validate_owner_repo(owner, repo):
            logger.error(f"Invalid owner/repo extracted from URL: {url}")
            return []

        logger.info(f"Creating GithubRepositoryReader for {owner}/{repo}")
        loader = GithubRepositoryReader(
            github_client,
            owner=owner,
            repo=repo,
            filter_file_extensions=(
                [".py", ".js", ".ts", ".md"],
                GithubRepositoryReader.FilterType.INCLUDE,
            ),
            verbose=False,
            concurrent_requests=5,
        )

        logger.info(f"Loading documents from branch: {branch}")
        docs = loader.load_data(branch=branch)
        logger.info(f"Successfully loaded {len(docs)} documents")

        for i, doc in enumerate(docs):
            logger.debug(f"Document {i+1}: {doc.metadata}")

        logger.info("Initializing Python semantic chunker")
        python_chunker = PythonSemanticChunker(max_chunk_size=2000, overlap_lines=5)

        def get_langchain_splitter(file_path: str):
            logger.debug(f"Getting splitter for file: {file_path}")
            if file_path.endswith(".ts"):
                logger.debug("Using TypeScript splitter")
                return RecursiveCharacterTextSplitter.from_language(
                    language=Language.TS, chunk_size=1000, chunk_overlap=100
                )
            elif file_path.endswith((".md", ".txt")):
                logger.debug("Using Markdown splitter")
                return RecursiveCharacterTextSplitter.from_language(
                    language=Language.MARKDOWN, chunk_size=1000, chunk_overlap=200
                )
            else:
                logger.debug("Using default splitter")
                return RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

        logger.info("Starting document chunking process")
        all_chunks: List[Dict[str, Any]] = []

        for i, doc in enumerate(docs):
            file_path = doc.metadata.get("file_path", "")
            logger.debug(f"Processing document {i+1}/{len(docs)}: {file_path}")

            if file_path.endswith(".py"):
                logger.debug("Using semantic chunking for Python file")
                semantic_chunks = python_chunker.extract_semantic_chunks(doc.text, file_path)
                logger.debug(f"Generated {len(semantic_chunks)} semantic chunks")

                for chunk_info in semantic_chunks:
                    metadata = doc.metadata.copy()
                    metadata["repo_name"] = repo
                    metadata["chunk_type"] = chunk_info["chunk_type"]
                    metadata["chunk_name"] = chunk_info["chunk_name"]
                    metadata["start_line"] = chunk_info["start_line"]
                    metadata["end_line"] = chunk_info["end_line"]
                    metadata["signature"] = chunk_info["signature"]
                    metadata["docstring"] = chunk_info["docstring"]
                    metadata["line_count"] = chunk_info["line_count"]
                    metadata["repo_link"] = url
                    

                    all_chunks.append({
                        "text": chunk_info["text"],
                        "metadata": metadata
                    })
            else:
                logger.debug("Using standard text splitter")
                splitter = get_langchain_splitter(file_path)
                chunks = splitter.split_text(doc.text)
                logger.debug(f"Generated {len(chunks)} standard chunks")

                for chunk in chunks:
                    metadata = doc.metadata.copy()
                    metadata["repo_name"] = repo
                    all_chunks.append({
                        "text": chunk,
                        "metadata": metadata
                    })

        logger.info(f"Chunking completed. Total chunks created: {len(all_chunks)}")
        return all_chunks

    except Exception as e:
        logger.error(f"Error loading GitHub repository: {e}", exc_info=True)
        return []


def convert_chunks_to_langchain_docs(all_chunks: List[Dict[str, Any]]) -> List[Document]:
    """Convert raw chunk dicts to LangChain Document objects."""
    logger.info(f"Converting {len(all_chunks)} chunks to LangChain Documents")
    langchain_docs = [
        Document(page_content=chunk["text"], metadata=chunk["metadata"])
        for chunk in all_chunks
    ]
    logger.info(f"Successfully converted {len(langchain_docs)} chunks to LangChain Documents")
    return langchain_docs

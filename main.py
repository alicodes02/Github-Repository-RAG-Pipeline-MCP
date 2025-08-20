#necessary imports
import nest_asyncio
import os
import textwrap
from dotenv import load_dotenv
from llama_index.readers.github import GithubRepositoryReader, GithubClient
from llama_index.core import download_loader
from llama_index.core import VectorStoreIndex
from langchain_text_splitters import (
    Language,
    RecursiveCharacterTextSplitter,
)
from pyhton_chunker import PythonSemanticChunker #local python class for chunking python files
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document
from langchain.vectorstores import Chroma

from groq import Groq
from sentence_transformers import CrossEncoder
import re

# Import logging configuration
from rag_mcp.logging_config import setup_logging, get_logger

# Initialize logging only once at the start
logger = setup_logging()

nest_asyncio.apply()
load_dotenv()

logger.info("Application started - main.py")
logger.info("Loading environment variables and initializing components")

# parse github url to match the pattern
def parse_github_url(url):
    logger.debug(f"Parsing GitHub URL: {url}")
    pattern = r"https://github\.com/([^/]+)/([^/]+)"
    match = re.match(pattern, url)
    result = match.groups() if match else (None, None)
    logger.debug(f"Parsed result: owner={result[0]}, repo={result[1]}")
    return result

# parse github url to extract owner and repo
def validate_owner_repo(owner, repo):
    logger.debug(f"Validating owner: {owner}, repo: {repo}")
    is_valid = bool(owner) and bool(repo)
    logger.debug(f"Validation result: {is_valid}")
    return is_valid

#initialize github client
def initialize_github_client():
    logger.info("Initializing GitHub client")
    github_token = os.getenv("GITHUB_TOKEN")
    if not github_token:
        logger.warning("GitHub token not found in environment variables")
    else:
        logger.info("GitHub token found, creating client")
    return GithubClient(github_token)

# Check for GitHub Token
github_token = os.getenv("GITHUB_TOKEN")
logger.info(f"GitHub token status: {'Found' if github_token else 'Not found'}")

# function to load GitHub repository and form chunks
def load_github_repository(url,branch):
    logger.info(f"Starting GitHub repository loading - URL: {url}, Branch: {branch}")

    try:
        github_client = initialize_github_client()
        #download_loader("GithubRepositoryReader")

        github_url = url
        branch_name = branch

        owner, repo = parse_github_url(github_url)

        if validate_owner_repo(owner, repo):
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
            logger.info(f"Loading {repo} repository by {owner}")
        else:
            logger.error(f"Invalid owner/repo extracted from URL: {url}")
            return []

        logger.info(f"Loading documents from branch: {branch_name}")
        docs = loader.load_data(branch=branch_name)
        logger.info(f"Successfully loaded {len(docs)} documents")
        
        for i, doc in enumerate(docs):
            logger.debug(f"Document {i+1}: {doc.metadata}")
        
        logger.info("Initializing Python semantic chunker")
        python_chunker = PythonSemanticChunker(max_chunk_size=2000, overlap_lines=5)

        #langchain splitters
        def get_langchain_splitter(file_path):
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

        # --- Chunking without node creation ---
        logger.info("Starting document chunking process")
        all_chunks = []

        for i, doc in enumerate(docs):
            file_path = doc.metadata.get("file_path", "")
            logger.debug(f"Processing document {i+1}/{len(docs)}: {file_path}")

            if file_path.endswith(".py"):
                logger.debug("Using semantic chunking for Python file")
                # Use semantic chunking for Python files
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

logger.info("Initializing embedding model")
embedding_model = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-en-v1.5",  
)
logger.info("Embedding model initialized successfully")

def convert_chunks_to_langchain_docs(all_chunks):
    logger.info(f"Converting {len(all_chunks)} chunks to LangChain Documents")

    # Convert chunks into LangChain Document format
    langchain_docs = [
        Document(page_content=chunk["text"], metadata=chunk["metadata"])
        for chunk in all_chunks
    ]

    logger.info(f"Successfully converted {len(langchain_docs)} chunks to LangChain Documents")
    return langchain_docs

def get_project_db_path():
    """Get the absolute path to the ChromaDB in the project root directory"""
    # Get the directory where main.py is located (project root)
    project_root = os.path.dirname(os.path.abspath(__file__))
    default_db_path = os.path.join(project_root, "chroma_db_hf")
    
    # Check for environment variable first
    db_path = os.getenv("DB_PATH", default_db_path)
    
    # If relative path, make it relative to project root
    if not os.path.isabs(db_path):
        db_path = os.path.join(project_root, db_path)
    
    return os.path.abspath(db_path)

def create_vector_store(langchain_docs, persist_directory=None):
    logger.info(f"Creating vector store with {len(langchain_docs)} documents")
    
    # Set directory for persistent ChromaDB storage
    if persist_directory is None:
        persist_directory = get_project_db_path()
    
    logger.info(f"Using persist directory: {persist_directory}")

    try:
        # Store documents and embeddings into ChromaDB
        logger.debug("Initializing ChromaDB")
        chroma_db = Chroma(
            embedding_function=embedding_model, 
            persist_directory=persist_directory
        )

        logger.info("Adding documents to ChromaDB")
        chroma_db.add_documents(langchain_docs)

        # Persist to disk
        logger.debug("Persisting ChromaDB to disk")
        chroma_db.persist()
        logger.info(f"Chunks added successfully to ChromaDB")
        return chroma_db
    except Exception as e:
        logger.error(f"Error creating vector store: {e}", exc_info=True)
        raise

def load_or_create_vector_store(langchain_docs=None, force_recreate=False, persist_directory=None, add_to_existing=False):
    """
    Load existing ChromaDB or create a new one if it doesn't exist.
    If add_to_existing is True, add new documents to existing database.
    """
    logger.info("Loading or creating vector store")
    
    if persist_directory is None:
        persist_directory = get_project_db_path()
    
    logger.info(f"Target persist directory: {persist_directory}")
    
    # Check if ChromaDB already exists
    if os.path.exists(persist_directory) and not force_recreate:
        logger.info(f"Found existing ChromaDB at {persist_directory}")
        try:
            # Load existing ChromaDB
            logger.debug("Attempting to load existing ChromaDB")
            chroma_db = Chroma(
                embedding_function=embedding_model,
                persist_directory=persist_directory
            )
            
            # Check if the database has documents
            collection = chroma_db._collection
            doc_count = collection.count()
            logger.info(f"Existing ChromaDB contains {doc_count} documents")
            
            if doc_count > 0:
                logger.info(f"Successfully loaded ChromaDB with {doc_count} documents")
                
                # If we want to add new documents to existing database
                if add_to_existing and langchain_docs:
                    logger.info(f"Adding {len(langchain_docs)} new documents to existing database")
                    chroma_db.add_documents(langchain_docs)
                    chroma_db.persist()
                    
                    # Check updated count
                    updated_count = chroma_db._collection.count()
                    logger.info(f"Database now contains {updated_count} documents (added {updated_count - doc_count} new documents)")
                
                return chroma_db
            else:
                logger.warning("ChromaDB exists but is empty. Creating new database...")
                if langchain_docs is None:
                    logger.error("No documents provided to populate empty database")
                    raise ValueError("No documents provided to populate empty database")
        except Exception as e:
            logger.error(f"Error loading ChromaDB: {e}", exc_info=True)
            logger.info("Creating new database...")
    
    # Create new ChromaDB if it doesn't exist or if force_recreate is True
    if langchain_docs is None:
        logger.error("No documents provided to create new database")
        raise ValueError("No documents provided to create new database")
    
    if force_recreate and os.path.exists(persist_directory):
        import shutil
        logger.warning(f"Force recreate enabled. Removing existing ChromaDB at {persist_directory}")
        shutil.rmtree(persist_directory)
    
    logger.info(f"Creating new ChromaDB at {persist_directory}")
    chroma_db = Chroma(
        embedding_function=embedding_model,
        persist_directory=persist_directory
    )
    
    logger.info(f"Adding {len(langchain_docs)} documents to ChromaDB...")
    chroma_db.add_documents(langchain_docs)
    
    # Persist to disk
    logger.debug("Persisting new ChromaDB to disk")
    chroma_db.persist()
    logger.info(f"ChromaDB created successfully with {len(langchain_docs)} documents")
    
    return chroma_db

# Initialize Groq client
logger.info("Initializing Groq client")
client = Groq(api_key=os.getenv("GROQ_API_KEY"))
if os.getenv("GROQ_API_KEY"):
    logger.info("Groq client initialized successfully")
else:
    logger.warning("GROQ_API_KEY not found - LLM responses will be unavailable")

# Load the bge reranker model
logger.info("Loading BGE reranker model")
reranker = CrossEncoder("BAAI/bge-reranker-large")
logger.info("BGE reranker model loaded successfully")

def build_context(chunks):
    logger.debug(f"Building context from {len(chunks)} chunks")
    context_texts = [doc.page_content for doc in chunks]
    context = "\n\n".join(context_texts)
    logger.debug(f"Context built with {len(context)} characters")
    return context

def ask_llm_with_context(question, chunks):
    logger.info(f"Asking LLM with context - Question: '{question}' using {len(chunks)} chunks")

    try:
        context = build_context(chunks)
        
        prompt = f"""
You are a helpful AI coding assistant whose task is to answer user queries from the context (usually code snippets or markdown files). Use the following context from the codebase to answer the user's question. Always use the context
to provide accurate and relevant answers. If sufficient context is not available, state that you cannot answer based on the provided information.

Context:
{context}

Question: {question}

Answer:"""

        logger.debug("Sending request to Groq API")
        response = client.chat.completions.create(
            #use updated llama3 model
            model="meta-llama/llama-4-scout-17b-16e-instruct",
            messages=[
                {"role": "system", "content": "You are a code expert assistant."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2
        )

        result = response.choices[0].message.content.strip()
        logger.info("LLM response received successfully")
        logger.debug(f"Response length: {len(result)} characters")
        return result
    except Exception as e:
        logger.error(f"Error getting LLM response: {e}", exc_info=True)
        return f"Error generating response: {str(e)}"

def query_chroma_db(chroma_db):
    logger.info("Starting interactive ChromaDB query session")

    while True:
        query = input("Your question: ")
        if query.lower() in ["exit", "quit"]:
            logger.info("User requested exit from query session")
            print("Goodbye! 👋")
            break

        logger.info(f"Processing user query: '{query}'")
        initial_results = chroma_db.similarity_search(query, k=15)
        logger.debug(f"Initial similarity search returned {len(initial_results)} results")

        print("Fetching relevant chunks from ChromaDB...")

        if not initial_results:
            logger.warning("No relevant chunks found for query")
            print("❌ No relevant chunks found.")
            continue

        print("Reranking chunks for relevance...")
        logger.info("Starting reranking process")

        # Prepare pairs for reranking: (query, chunk)
        rerank_inputs = [[query, doc.page_content] for doc in initial_results]
        logger.debug(f"Prepared {len(rerank_inputs)} pairs for reranking")
        
        # Get relevance scores from the reranker
        scores = reranker.predict(rerank_inputs)
        logger.debug("Reranking scores computed")

        # Rank and select top 4 most relevant chunks
        ranked_docs = sorted(zip(initial_results, scores), key=lambda x: x[1], reverse=True)
        top_docs = [doc for doc, score in ranked_docs[:4]]
        logger.info(f"Selected top {len(top_docs)} chunks after reranking")
        
        print("\n🔎 Top matching chunks:")
        for i, doc in enumerate(top_docs, 1):
            print(f"\nResult {i}:")
            # get file path and repo name from metadata
            print("File:", doc.metadata.get("file_path", "N/A"))
            print("Repo:", doc.metadata.get("repo_name", "N/A"))
            print(doc.page_content) 

        print("\n🧠 Sending context to Groq LLM...\n")
        logger.info("Query processing completed")

def query_chroma_with_hardcoded(chroma_db, query):
    """
    Query ChromaDB with a specific query and return the top reranked chunks.
    """
    logger.info(f"Processing hardcoded query: '{query}'")
    print(f"\n📝 Querying ChromaDB with: '{query}'")
    print("=" * 60)
    
    try:
        # Perform similarity search
        logger.debug("Performing similarity search")
        initial_results = chroma_db.similarity_search(query, k=15)
        
        if not initial_results:
            logger.warning("No relevant chunks found")
            print("❌ No relevant chunks found.")
            return []
        
        logger.info(f"Found {len(initial_results)} initial chunks")
        print(f"✅ Found {len(initial_results)} initial chunks")
        print("\n🔄 Reranking chunks for relevance...")
        
        # Prepare pairs for reranking: (query, chunk)
        logger.debug("Preparing reranking inputs")
        rerank_inputs = [[query, doc.page_content] for doc in initial_results]
        
        # Get relevance scores from the reranker
        logger.debug("Computing reranking scores")
        scores = reranker.predict(rerank_inputs)
        
        # Rank and select top 4 most relevant chunks
        logger.debug("Sorting and selecting top chunks")
        ranked_docs = sorted(zip(initial_results, scores), key=lambda x: x[1], reverse=True)
        top_docs = ranked_docs[:4]
        
        logger.info(f"Selected top {len(top_docs)} chunks")
        print(f"\n🎯 Top {len(top_docs)} most relevant chunks:")
        print("=" * 60)
        
        for i, (doc, score) in enumerate(top_docs, 1):
            logger.debug(f"Displaying chunk {i} with score {score:.4f}")
            print(f"\n📄 Chunk {i} (Relevance Score: {score:.4f})")
            print("-" * 40)
            print(f"📁 File: {doc.metadata.get('file_path', 'N/A')}")
            print(f"📦 Repo: {doc.metadata.get('repo_name', 'N/A')}")
            print(f"\n📝 Content Preview (first 500 chars):")
            print(textwrap.fill(doc.page_content[:500], width=80))
            if len(doc.page_content) > 500:
                print(f"... [{len(doc.page_content) - 500} more characters]")
            print("-" * 40)
        
        print("\n✅ Query completed successfully!")
        logger.info("Hardcoded query completed successfully")
        return top_docs
    except Exception as e:
        logger.error(f"Error in hardcoded query: {e}", exc_info=True)
        print(f"❌ Error processing query: {e}")
        return []

if __name__ == "__main__":
    logger.info("Starting main application")

    print("Welcome to the GitHub Codebase Query Assistant! 🤖\n")

    # Check if ChromaDB already exists
    persist_directory = get_project_db_path()
    use_existing = False
    add_new_docs = False
    
    logger.debug(f"Checking for existing ChromaDB at {persist_directory}")
    if os.path.exists(persist_directory):
        logger.info(f"Found existing ChromaDB at {persist_directory}")
        print(f"📁 Found existing ChromaDB at {persist_directory}")
        user_choice = input("Do you want to:\n1. Use existing database (u)\n2. Add new repository to existing database (a)\n3. Create new database (n)\nChoice (u/a/n, default: u): ").strip().lower()
        
        if user_choice == 'a':
            use_existing = True
            add_new_docs = True
            logger.info("User chose to add new documents to existing database")
        elif user_choice == 'n':
            use_existing = False
            logger.info("User chose to create new database")
        else:
            use_existing = True
            logger.info("User chose to use existing database")
    
    if use_existing and os.path.exists(persist_directory) and not add_new_docs:
        logger.info("Loading existing ChromaDB")
        # Load existing ChromaDB without creating new documents
        chroma_db = load_or_create_vector_store(langchain_docs=None, force_recreate=False, persist_directory=persist_directory)
    else:
        logger.info("Creating new ChromaDB or adding to existing from GitHub repository")
        # Create new ChromaDB with documents from GitHub or add to existing
        print("\n🔄 Processing GitHub repository...\n")
        
        # Hardcoded GitHub repository URL and branch
        github_repo_url = "https://github.com/langchain-ai/langchain"  # Example repository
        branch_name = "master"
        
        logger.info(f"Target repository: {github_repo_url}, branch: {branch_name}")
        print(f"Loading repository: {github_repo_url}")
        print(f"Branch: {branch_name}\n")
        
        chunks = load_github_repository(url=github_repo_url, branch=branch_name)
        langchain_docs = convert_chunks_to_langchain_docs(all_chunks=chunks)
        
        if add_new_docs:
            # Add to existing database
            chroma_db = load_or_create_vector_store(
                langchain_docs=langchain_docs, 
                force_recreate=False, 
                persist_directory=persist_directory,
                add_to_existing=True
            )
        else:
            # Create or recreate the vector store
            chroma_db = load_or_create_vector_store(
                langchain_docs=langchain_docs, 
                force_recreate=True, 
                persist_directory=persist_directory
            )

    # Hardcoded query to ChromaDB
    query = input("Enter your query: ")
    logger.info(f"User entered query: '{query}'")
    
    # Call the separate function to query ChromaDB
    top_chunks = query_chroma_with_hardcoded(chroma_db, query)
    
    if top_chunks:
        logger.info(f"Successfully retrieved and ranked {len(top_chunks)} chunks")
        print(f"\n📊 Successfully retrieved and ranked {len(top_chunks)} chunks from ChromaDB")
    else:
        logger.warning("No chunks were retrieved from the query")
        print("\n⚠️ No chunks were retrieved from the query")
    
    logger.info("Main application completed")

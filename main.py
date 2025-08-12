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

nest_asyncio.apply()
load_dotenv()

# parse github url to match the pattern
def parse_github_url(url):
    pattern = r"https://github\.com/([^/]+)/([^/]+)"
    match = re.match(pattern, url)
    return match.groups() if match else (None, None)

# parse github url to extract owner and repo
def validate_owner_repo(owner, repo):
    return bool(owner) and bool(repo)

#initialize github client
def initialize_github_client():
    github_token = os.getenv("GITHUB_TOKEN")
    return GithubClient(github_token)

# Check for GitHub Token
github_token = os.getenv("GITHUB_TOKEN")
if not github_token:
    raise EnvironmentError("GitHub token not found in environment variables")

# function to load GitHub repository and form chunks
def load_github_repository(url,branch):

    github_client = initialize_github_client()
    download_loader("GithubRepositoryReader")

    github_url = url
    branch_name = branch

    owner, repo = parse_github_url(github_url)

    if validate_owner_repo(owner, repo):
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
        print(f"Loading {repo} repository by {owner}")

    docs = loader.load_data(branch=branch_name)
    print("Documents uploaded:")
    for doc in docs:
        print(doc.metadata)
    print(f"Total documents: {len(docs)}")

    # print("Uploading to vector store...")

    python_chunker = PythonSemanticChunker(max_chunk_size=2000, overlap_lines=5)

    #langchain splitters
    def get_langchain_splitter(file_path):
        # if file_path.endswith(".py"):
        #     return RecursiveCharacterTextSplitter.from_language(
        #         language=Language.PYTHON, chunk_size=1000, chunk_overlap=100
        #     )
        if file_path.endswith(".ts"):
            return RecursiveCharacterTextSplitter.from_language(
                language=Language.TS, chunk_size=1000, chunk_overlap=100
            )
        elif file_path.endswith((".md", ".txt")):
            return RecursiveCharacterTextSplitter.from_language(
                language=Language.MARKDOWN, chunk_size=1000, chunk_overlap=200
            )
        else:
            return RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

    # --- Chunking without node creation ---

    all_chunks = []

    for doc in docs:
        file_path = doc.metadata.get("file_path", "")

        if file_path.endswith(".py"):
            # Use semantic chunking for Python files
            semantic_chunks = python_chunker.extract_semantic_chunks(doc.text, file_path)
            
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
            splitter = get_langchain_splitter(file_path)
            chunks = splitter.split_text(doc.text)

            for chunk in chunks:
                metadata = doc.metadata.copy()
                metadata["repo_name"] = repo
                all_chunks.append({
                    "text": chunk,
                    "metadata": metadata
                })

    print(f"Total chunks created: {len(all_chunks)}")

    return all_chunks


embedding_model = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-en-v1.5",  
)

def convert_chunks_to_langchain_docs(all_chunks):

    # Convert chunks into LangChain Document format
    langchain_docs = [
        Document(page_content=chunk["text"], metadata=chunk["metadata"])
        for chunk in all_chunks
    ]

    return langchain_docs

def create_vector_store(langchain_docs):
    # Set directory for persistent ChromaDB storage
    persist_directory = "./chroma_db_hf"

    # Store documents and embeddings into ChromaDB
    chroma_db = Chroma(
        embedding_function=embedding_model, 
        persist_directory=persist_directory
    )

    chroma_db.add_documents(langchain_docs)

    # Persist to disk
    chroma_db.persist()
    print(f"Chunks added successfully.")

    return chroma_db

# Initialize Groq client
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# Load the bge reranker model
reranker = CrossEncoder("BAAI/bge-reranker-large")

def build_context(chunks):
    context_texts = [doc.page_content for doc in chunks]
    return "\n\n".join(context_texts)

def ask_llm_with_context(question, chunks):

    context = build_context(chunks)
    
    prompt = f"""
You are a helpful AI coding assistant whose task is to answer user queries from the context (usually code snippets or markdown files). Use the following context from the codebase to answer the user's question. Always use the context
to provide accurate and relevant answers. If sufficient context is not available, state that you cannot answer based on the provided information.

Context:
{context}

Question: {question}

Answer:"""

    response = client.chat.completions.create(
        #use updated llama3 model
        model="meta-llama/llama-4-scout-17b-16e-instruct",
        messages=[
            {"role": "system", "content": "You are a code expert assistant."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.2
    )

    return response.choices[0].message.content.strip()

def query_chroma_db(chroma_db):

    while True:
        query = input("Your question: ")
        if query.lower() in ["exit", "quit"]:
            print("Goodbye! 👋")
            break

        initial_results = chroma_db.similarity_search(query, k=15)

        print("Fetching relevant chunks from ChromaDB...")

        if not initial_results:
            print("❌ No relevant chunks found.")
            continue

        print("Reranking chunks for relevance...")

        # Prepare pairs for reranking: (query, chunk)
        rerank_inputs = [[query, doc.page_content] for doc in initial_results]
        
        # Get relevance scores from the reranker
        scores = reranker.predict(rerank_inputs)

        # Rank and select top 4 most relevant chunks
        ranked_docs = sorted(zip(initial_results, scores), key=lambda x: x[1], reverse=True)
        top_docs = [doc for doc, score in ranked_docs[:4]]
        
        print("\n🔎 Top matching chunks:")
        for i, doc in enumerate(top_docs, 1):
            print(f"\nResult {i}:")
            # get file path and repo name from metadata
            print("File:", doc.metadata.get("file_path", "N/A"))
            print("Repo:", doc.metadata.get("repo_name", "N/A"))
            print(doc.page_content) 

        print("\n🧠 Sending context to Groq LLM...\n")
        answer = ask_llm_with_context(query, top_docs)
        print("💡 LLM Answer:\n", answer)


if __name__ == "__main__":

    #load_dotenv()  # Load environment variables from .env file
    print("Welcome to the GitHub Codebase Query Assistant! 🤖")
    print("You can ask questions about the codebase, and I'll do my best to help you.\n")

    github_repo_url = input("Enter the GitHub repository URL")
    branch_name = input("Enter the branch name (default is 'dev'): ") or "dev"
    
    chunks = load_github_repository(url=github_repo_url, branch=branch_name)
    langchain_docs=convert_chunks_to_langchain_docs(all_chunks=chunks)

    chroma_db = create_vector_store(langchain_docs=langchain_docs)

    query_chroma_db(chroma_db=chroma_db)




    






    








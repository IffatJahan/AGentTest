# -----------------------------
# Manual RAG Pipeline without RetrievalQA
# -----------------------------
from dotenv import load_dotenv

# Load environment variables (e.g., API keys if needed)
load_dotenv()
from langchain_community.document_loaders  import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from langchain_ollama.llms import OllamaLLM
import os
# -----------------------------
# 1. Load PDF and split into chunks
# -----------------------------
pdf_path = r"E:\Doc\Bank Credit Rating AI Prompt\Bank Credit Rating AI Prompt\4. Bank Prompt.txt"
loader = TextLoader(pdf_path)
docs = loader.load()

splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = splitter.split_documents(docs)

# -----------------------------
# 2. Create embeddings + vector store
# -----------------------------
# Use a valid embedding model (nomic-embed-text may not exist)

embeddings = OllamaEmbeddings(model="llama3")
vector_store = Chroma.from_documents(chunks, embedding=embeddings)

# -----------------------------
# 3. Define retriever
# -----------------------------
def retrieve(query, k=5):
    """Retrieve top-k relevant chunks for a query"""
    return vector_store.similarity_search(query, k=k)


# -----------------------------
# 4. Define LLM
# -----------------------------
llm = OllamaLLM(model="llama3")  # or "mistral", "deepseek", etc.


# -----------------------------
# 5. Manual RAG function
# -----------------------------
def run_rag(query):
    retrieved_docs = retrieve(query)
    context = "\n\n".join([doc.page_content for doc in retrieved_docs])

    prompt = f"""
You are a professional financial analyst. Using ONLY the context below from the PDF, 
generate a summary for the following query:

Query:
{query}

Context:
{context}
"""
    answer = llm.invoke(prompt)
    return answer, retrieved_docs


# -----------------------------
# 6. Run query
# -----------------------------
query = "Summarize all financial values and trends in the report"
report, sources = run_rag(query)

print("=== Generated Report ===")
print(report)

print("\n=== Sources Used ===")
for doc in sources:
    page = doc.metadata.get("page_number", "N/A")
    print(f"- Page {page}: {doc.page_content[:200]}...")

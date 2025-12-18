from dotenv import load_dotenv

load_dotenv()

from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import numpy as np
from sentence_transformers import SentenceTransformer
import chromadb
import uuid
from typing import List, Dict, Any
import os

pdf_path = r"E:\Doc\Bank Credit Rating AI Prompt\Bank Credit Rating AI Prompt\\1. Annual_Report_2024.pdf"


def split_doc(doc, chunk_size=500, chunk_overlap=50):
    """Split doc into smaller chunks for better RAG performance"""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", " ", ""]
    )
    chunks = text_splitter.split_documents(doc)
    print(f"[INFO] Split {len(doc)} documents into {len(chunks)} chunks.")
    return chunks


def embed_chunks(texts: List[str]) -> np.ndarray:
    """Generate normalized embeddings for text chunks"""
    model = SentenceTransformer("all-MiniLM-L6-v2")
    print(f"[INFO] Generating embeddings for {len(texts)} chunks...")
    embeddings = model.encode(texts, show_progress_bar=True, normalize_embeddings=True)
    print(f"[INFO] Embeddings shape: {embeddings.shape}")
    return embeddings


def get_or_create_collection(collection_name: str = "pdf_documents",
                             persist_directory: str = "../data/vector_store"):
    """Get existing collection or create new one"""
    try:
        os.makedirs(persist_directory, exist_ok=True)
        client = chromadb.PersistentClient(path=persist_directory)

        collection = client.get_or_create_collection(
            name=collection_name,
            metadata={"description": "PDF document embeddings for RAG"}
        )
        print(f"[INFO] Collection '{collection_name}' loaded. Documents: {collection.count()}")
        return collection
    except Exception as e:
        print(f"[ERROR] Failed to get collection: {e}")
        raise


def add_documents_VectorStorage(documents: List[Any], embeddings: np.ndarray,
                                collection_name: str = "pdf_documents"):
    """Add documents and embeddings to vector store"""
    if len(documents) != len(embeddings):
        raise ValueError("Number of documents must match number of embeddings")

    collection = get_or_create_collection(collection_name)

    # Check if already populated
    if collection.count() > 0:
        print(f"[INFO] Collection already has {collection.count()} documents. Skipping add.")
        return collection

    print(f"[INFO] Adding {len(documents)} documents to vector store...")

    ids = []
    metadatas = []
    documents_text = []
    embeddings_list = []

    for i, (doc, embedding) in enumerate(zip(documents, embeddings)):
        doc_id = f"doc_{uuid.uuid4().hex[:8]}_{i}"
        ids.append(doc_id)

        metadata = dict(doc.metadata)
        metadata['doc_index'] = i
        metadata['content_length'] = len(doc.page_content)
        metadatas.append(metadata)

        documents_text.append(doc.page_content)
        embeddings_list.append(embedding.tolist())

    try:
        collection.add(
            ids=ids,
            embeddings=embeddings_list,
            metadatas=metadatas,
            documents=documents_text
        )
        print(f"[SUCCESS] Added {len(documents)} documents. Total: {collection.count()}")
        return collection
    except Exception as e:
        print(f"[ERROR] Failed to add documents: {e}")
        raise


def RAGRetriever(query: str, top_k: int = 5, score_threshold: float = 0.3,
                 collection_name: str = "pdf_documents") -> List[Dict[str, Any]]:
    """Retrieve relevant documents for a query"""
    print(f"\n[RETRIEVAL] Query: '{query}'")
    print(f"[RETRIEVAL] Top K: {top_k}, Score threshold: {score_threshold}")

    try:
        # Get collection
        collection = get_or_create_collection(collection_name)

        if collection.count() == 0:
            print("[WARNING] Collection is empty!")
            return []

        # Generate query embedding
        query_embedding = embed_chunks([query])[0]

        # Search in vector store
        results = collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=min(top_k, collection.count())
        )

        # Process results
        retrieved_docs = []

        if results['documents'] and results['documents'][0]:
            documents = results['documents'][0]
            metadatas = results['metadatas'][0]
            distances = results['distances'][0]
            ids = results['ids'][0]

            for i, (doc_id, document, metadata, distance) in enumerate(
                    zip(ids, documents, metadatas, distances)
            ):
                # Convert distance to similarity score
                similarity_score = 1 - distance

                if similarity_score >= score_threshold:
                    retrieved_docs.append({
                        'id': doc_id,
                        'content': document,
                        'metadata': metadata,
                        'similarity_score': similarity_score,
                        'distance': distance,
                        'rank': i + 1
                    })
                    print(f"  [DOC {i + 1}] Score: {similarity_score:.4f} | Preview: {document[:100]}...")

            print(f"[SUCCESS] Retrieved {len(retrieved_docs)} documents (threshold: {score_threshold})")
        else:
            print("[WARNING] No documents found in results")

        return retrieved_docs

    except Exception as e:
        print(f"[ERROR] Retrieval failed: {e}")
        import traceback
        traceback.print_exc()
        return []


def Chatbot(query: str, collection_name: str = "pdf_documents"):
    """Generate answer using RAG"""
    print(f"\n{'=' * 60}")
    print(f"CHATBOT QUERY: {query}")
    print(f"{'=' * 60}")

    # Retrieve relevant documents
    results = RAGRetriever(query, top_k=5, score_threshold=0.3,
                           collection_name=collection_name)

    if not results:
        return "❌ No relevant context found to answer the question. Try lowering the score threshold or check if documents are properly indexed."

    # Build context from retrieved documents
    context = "\n\n".join([
        f"[Document {doc['rank']}] (Score: {doc['similarity_score']:.4f})\n{doc['content']}"
        for doc in results
    ])

    print(f"\n[CONTEXT] Using {len(results)} documents for answer generation")

    try:
        # Initialize LLM
        model = HuggingFaceEndpoint(
            repo_id="deepseek-ai/DeepSeek-V3.1-Terminus",
            task="text-generation",
            max_new_tokens=512,
            do_sample=False,
            repetition_penalty=1.03,
        )
        llm = ChatHuggingFace(llm=model)

        # Create prompt
        prompt = f"""Use the following context to answer the question concisely and accurately.

Context:
{context}

Question: {query}

Answer:"""

        # Generate response
        print("[LLM] Generating answer...")
        response = llm.invoke(prompt)
        answer = response.content

        print(f"\n[ANSWER]\n{answer}")
        print(f"{'=' * 60}\n")

        return answer

    except Exception as e:
        print(f"[ERROR] Answer generation failed: {e}")
        import traceback
        traceback.print_exc()
        return f"Error generating answer: {str(e)}"


# ============= MAIN EXECUTION =============
if __name__ == "__main__":
    COLLECTION_NAME = "pdf_documents"

    # Load and process documents (only if collection is empty)
    collection = get_or_create_collection(COLLECTION_NAME)

    if collection.count() == 0:
        print("\n[SETUP] Loading and processing PDF...")
        loader = PyMuPDFLoader(pdf_path)
        doc = loader.load()

        # Split into chunks
        chunks = split_doc(doc, chunk_size=500, chunk_overlap=50)

        # Generate embeddings
        texts = [chunk.page_content for chunk in chunks]
        embeddings = embed_chunks(texts)

        # Store in vector database
        add_documents_VectorStorage(chunks, embeddings, COLLECTION_NAME)
    else:
        print(f"\n[SETUP] Using existing collection with {collection.count()} documents")

    # Test chatbot
    print("\n" + "=" * 60)
    print("TESTING CHATBOT")
    print("=" * 60)

    # Test queries
    test_queries = [
        "What is the Record Date?",
        "What is the company's revenue?",
        "Who is the CEO?"
    ]

    for query in test_queries:
        answer = Chatbot(query, COLLECTION_NAME)
        print()  # Spacing between queries
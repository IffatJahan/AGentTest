from dotenv import load_dotenv
# Load environment variables (e.g., API keys if needed)
load_dotenv()
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import TextLoader,PyPDFLoader,PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import numpy as np
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings
import uuid
from typing import List, Dict, Any, Tuple
from sklearn.metrics.pairwise import cosine_similarity
import os

pdf_path1 = r"E:\Doc\Bank Credit Rating AI Prompt\Bank Credit Rating AI Prompt\\4. Bank Prompt.txt"
pdf_path = r"E:\Doc\Bank Credit Rating AI Prompt\Bank Credit Rating AI Prompt\\1. Annual_Report_2024.pdf"
loader=PyMuPDFLoader(pdf_path)
collectionMain=None
doc=loader.load()
# for d in doc:
#     d.metadata['source_file']='1. Annual_Report_2024.pdf'
# print(doc[0].metadata)

def split_doc(doc,chunk_size=500,chunk_overlap=50):
    """Split doc into smaller chunks fro better RAG performance"""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n","\n"," ",""]
    )
    chunks = text_splitter.split_documents(doc)
    print(f"[INFO] Split {len(doc)} documents into {len(chunks)} chunks.")
    return chunks;

def embed_chunks(texts: List[str]) -> np.ndarray:
        model = SentenceTransformer("all-MiniLM-L6-v2")
        # print(f"[INFO] Generating embeddings for {len(texts)} chunks...")
        embeddings = model.encode(texts, show_progress_bar=True, normalize_embeddings=True)
        # print(f"[INFO] Embeddings shape: {embeddings.shape}")
        return embeddings;


def initialize_store(collection_name: str = "pdf_documents", persist_directory: str = "../data/vector_store"):
    """Initialize ChromaDB client and collection"""
    try:
        # Create persistent ChromaDB client
        os.makedirs(persist_directory, exist_ok=True)
        client = chromadb.PersistentClient(path=persist_directory)

        # Get or create collection
        collection = client.get_or_create_collection(
            name=collection_name,
            metadata={"description": "PDF document embeddings for RAG"}
        )
        print(f"Vector store initialized. Collection: {collection_name}")
        print(f"Existing documents in collection: {collection.count()}")

    except Exception as e:
        print(f"Error initializing vector store: {e}")
        raise
    return collection

def Get_store(collection_name: str = "pdf_documents", persist_directory: str = "../data/vector_store"):
    """Initialize ChromaDB client and collection"""
    try:
        # Create persistent ChromaDB client
        client = chromadb.PersistentClient(path=persist_directory)
        # Get or create collection
        collection = client.get_collection(
            name=collection_name

        )

    except Exception as e:
        print(f"Error initializing vector store: {e}")
        raise
    return collection
def add_documents_VectorStorage(documents: List[Any], embeddings: np.ndarray):

        collection = initialize_store()
        """
        Add documents and their embeddings to the vector store

        Args:
            documents: List of LangChain documents
            embeddings: Corresponding embeddings for the documents
        """
        if len(documents) != len(embeddings):
            raise ValueError("Number of documents must match number of embeddings")

        print(f"Adding {len(documents)} documents to vector store...")

        # Prepare data for ChromaDB
        ids = []
        metadatas = []
        documents_text = []
        embeddings_list = []

        for i, (doc, embedding) in enumerate(zip(documents, embeddings)):
            # Generate unique ID
            doc_id = f"doc_{uuid.uuid4().hex[:8]}_{i}"
            ids.append(doc_id)

            # Prepare metadata
            metadata = dict(doc.metadata)
            metadata['doc_index'] = i
            metadata['content_length'] = len(doc.page_content)
            metadatas.append(metadata)

            # Document content
            documents_text.append(doc.page_content)

            # Embedding
            embeddings_list.append(embedding.tolist())

        # Add to collection
        try:
            collection.add(
                ids=ids,
                embeddings=embeddings_list,
                metadatas=metadatas,
                documents=documents_text
            )
            print(f"Successfully added {len(documents)} documents to vector store")
            print(f"Total documents in collection: {collection.count()}")

        except Exception as e:
            print(f"Error adding documents to vector store: {e}")
            raise

def RAGRetriever (query: str, top_k: int = 5, score_threshold: float = 0.0) -> List[Dict[str, Any]]:

    """
           Retrieve relevant documents for a query

           Args:
               query: The search query
               top_k: Number of top results to return
               score_threshold: Minimum similarity score threshold

           Returns:
               List of dictionaries containing retrieved documents and metadata
           """
    print(f"Retrieving documents for query: '{query}'")
    print(f"Top K: {top_k}, Score threshold: {score_threshold}")
    # Generate query embedding
    queryEmbedding = embed_chunks([query])[0]
    # print(f"query embedded {queryEmbedding} ")
    # Search in vector store
    try:
        store = Get_store()
        print(store.count())
        results = store.query(
            query_embeddings=[queryEmbedding.tolist()],
            n_results=top_k
        )

        # Process results
        retrieved_docs = []

        if results['documents'] and results['documents'][0]:
            documents = results['documents'][0]
            metadatas = results['metadatas'][0]
            distances = results['distances'][0]
            ids = results['ids'][0]

            for i, (doc_id, document, metadata, distance) in enumerate(zip(ids, documents, metadatas, distances)):
                # Convert distance to similarity score (ChromaDB uses cosine distance)
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

            print(f"Retrieved {len(retrieved_docs)} documents (after filtering)")
        else:
            print("No documents found")

        return retrieved_docs

    except Exception as e:
        print(f"Error during retrieval: {e}")
        return []

    except Exception as e:
        print(f"Error during retrieval: {e}")
        return []


def Chatbot(query):
    results=RAGRetriever(query)
    model = HuggingFaceEndpoint(
        repo_id="deepseek-ai/DeepSeek-V3.1-Terminus",
        task="text-generation",
        max_new_tokens=512,
        do_sample=False,
        repetition_penalty=1.03,
        provider="auto",  # let Hugging Face choose the best provider for you
    )

    llm = ChatHuggingFace(llm=model)
    """
           Retrieve relevant documents for a query

           Args:
               query: The search query
               top_k: Number of top results to return
               score_threshold: Minimum similarity score threshold

           Returns:
               List of dictionaries containing retrieved documents and metadata
           """
    try:

        # Process results
        context = "\n\n".join([doc['content'] for doc in results]) if results else ""
        if not context:
            return "No relevant context found to answer the question."

        ## generate the answwer using GROQ LLM
        prompt = f"""Use the following context to answer the question concisely.
                Context:
                {context}

                Question: {query}

                Answer:"""

        response = llm.invoke([prompt.format(context=context, query=query)])
        print(f"Finel result {response.content}")
        return response.content;

    except Exception as e:
        print(f"Error during retrieval: {e}")
        return []


chunks = split_doc(doc)
### Convert the text to embeddings
texts = [chunk.page_content for chunk in chunks]
## Generate the Embeddings
embeddings = embed_chunks(texts)
print("[INFO] Example embedding:", embeddings[0] if len(embeddings) > 0 else None)
##store int he vector dtaabase
add_documents_VectorStorage(chunks,embeddings)
# call chatbot
Chatbot("What is the Record Date?")

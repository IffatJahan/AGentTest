
import os
from dotenv import load_dotenv

# Load environment variables (e.g., API keys if needed)
load_dotenv()

from langchain_community.document_loaders import PyPDFLoader,TextLoader,PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import Ollama
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_classic.chains import RetrievalQA
from langchain_openai import ChatOpenAI


# 1. Load and chunk PDF
pdf_path = r"E:\Doc\Bank Credit Rating AI Prompt\Bank Credit Rating AI Prompt\\Best Holdings PLC.pdf"
# pdf_path = r"report_text.txt"
loader = PyMuPDFLoader(pdf_path)
docs = loader.load()
for d in docs:
    d.page_content='File Name: Best Holdings PLC.pdf'+d.page_content
# print(docs[0].metadata)
splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=300)
chunks = splitter.split_documents(docs)

# 2. Create embeddings + vector DB
# embeddings = OllamaEmbeddings(model="deepseek-v3.1:671b-cloud")  # embedding model
# embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
embeddings = HuggingFaceEmbeddings(model_name="nomic-ai/nomic-embed-text-v1.5",  model_kwargs={
        "trust_remote_code": True  # important!
    })

# vectorstore = Chroma.from_documents(chunks, embedding=embeddings, persist_directory="./vectorstore" )
vectorstore = Chroma.from_documents(chunks, embedding=embeddings, persist_directory="./vectorstore_1" )
# Query the Vector Store


# 3. Define retriever
retriever = vectorstore.as_retriever(search_type="similarity",search_kwargs={"k": 10})

# 4. Define LLM (via Ollama)
# llm = Ollama(model="llama3")  # swap with mistral, deepseek, etc.
llm = HuggingFaceEndpoint(
    repo_id="deepseek-ai/DeepSeek-V3.1-Terminus",
    # repo_id="meta-llama/Llama-3.1-8B-Instruct",
    task="text-generation",
    max_new_tokens=512,
    do_sample=False,
    repetition_penalty=1.03,
    provider="auto",  # let Hugging Face choose the best provider for you
)
# model =  ChatOpenAI(model="gpt-5")
model = ChatHuggingFace(llm=llm)


# 5. Build RAG chain
qa_chain = RetrievalQA.from_chain_type(
    llm=model,
    retriever=retriever,
    chain_type="stuff",   # simplest: stuff retrieved docs into prompt
    return_source_documents=True,

)
while True:
    query = input("\nAsk something (or type 'exit'): ")
    results = vectorstore.similarity_search(
        query,
        k=10
    )
    context = "\n\n".join([doc.page_content for doc in results]) if results else ""
    if not context:
        print( "No relevant context found to answer the question.")

    ## generate the answwer using GROQ LLM
    prompt = f""" don't use this format as a context and Use the following context to answer the question concisely.

                    Context:
                   
                    {context}

                    Question: 
                    
                    {query}

                    Answer:"""
    if query.lower() == "exit":
        break

    result = qa_chain.invoke(prompt)
    print("\n--- Answer ---")
    print(result["result"])
# 6. Run query
# query = "What is the file name?"
# result = qa_chain.invoke(query)
#
# print("=== Generated Report ===")
# print(result["result"])

# print("\n=== Sources Used ===")
# for doc in result["source_documents"]:
#     print(f"- Page {doc.metadata.get('page_number')}: {doc.page_content[:200]}...")



# 📦 Requirements Recap
# Here’s the requirements list aligned with this script:
# langchain~=1.0.7
# langchain-community~=0.4.1
# langgraph~=1.0.3
# python-dotenv~=1.2.1
# requests~=2.32.5
# openai~=1.109.1
# chromadb~=0.4.22
# reportlab~=4.0.9
# ollama~=0.1.7
# pypdf~=4.0.1
#
#
#
# 🔧 Notes
# - langchain-community is required for Ollama, Chroma, and loaders.
# - langchain provides chains, text splitters, and output parsers.
# - python-dotenv lets you manage environment variables (e.g., API keys).
# - reportlab is optional if you want to generate structured PDF outputs later.
# - openai is optional unless you plan to call GPT models via API.
#
# 👉 This script is now fully aligned with your requirements and the latest LangChain v1.x structure.
# Would you like me to extend this so it writes the generated financial summary into a new PDF using reportlab, so you get a finished report file instead of just console output?

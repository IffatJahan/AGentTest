from typing import Annotated
from dotenv import load_dotenv
from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.graph import StateGraph, START, END
load_dotenv()
from typing import Annotated, Any, Dict, Optional, TypedDict
from langgraph.graph import StateGraph, START, END
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import InMemorySaver
from langchain_community.tools import DuckDuckGoSearchRun
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.tools import tool
from langchain_community.document_loaders import PyPDFLoader,PyMuPDFLoader
import tempfile
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores import Chroma
import os
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
import base64
from langsmith import traceable
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama

#======================== Model Setup =======================================
class ChatState(TypedDict):

    messages: Annotated[list[BaseMessage], add_messages]

llm = HuggingFaceEndpoint(
    repo_id="deepseek-ai/DeepSeek-V3.1-Terminus",
    # repo_id="meta-llama/Llama-3.1-8B-Instruct",
    task="text-generation",
    max_new_tokens=512,
    do_sample=False,
    repetition_penalty=1.03,
    # provider="auto",  # let Hugging Face choose the best provider for you
)

#
# model = ChatOllama(
#     model="gpt-oss:20b-cloud",
#     temperature=0,
#     # other params...
# )
model =  ChatOpenAI(model="gpt-5")
# model = ChatHuggingFace(llm=llm)

embeddings = HuggingFaceEmbeddings(model_name="nomic-ai/nomic-embed-text-v1.5",  model_kwargs={
        "trust_remote_code": True  # important!
    })
#========================== Tools==========================================
# -------------------
# 2. PDF retriever store (per thread)
# -------------------
_THREAD_RETRIEVERS: Dict[str, Any] = {}
_THREAD_METADATA: Dict[str, dict] = {}

def _get_retriever(thread_id: Optional[str]):
    """Fetch the retriever for a thread if available."""
    if thread_id and thread_id in _THREAD_RETRIEVERS:
        return _THREAD_RETRIEVERS[thread_id]
    return None


def ingest_pdf(file_bytes: bytes, thread_id: str, filename: Optional[str] = None) -> dict:
    """
    Build a FAISS retriever for the uploaded PDF and store it for the thread.

    Returns a summary dict that can be surfaced in the UI.
    """
    if not file_bytes:
        raise ValueError("No bytes received for ingestion.")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
        temp_file.write(file_bytes)
        temp_path = temp_file.name

    try:
        loader = PyMuPDFLoader(temp_path)
        docs = loader.load()

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=2000, chunk_overlap=200, separators=["\n\n", "\n", " ", ""]
        )
        chunks = splitter.split_documents(docs)

        vector_store = Chroma.from_documents(chunks,
                                             embeddings,
                                             persist_directory="./vectorstore_UI" ,
                                             collection_name=thread_id,
                                             # metadatas=[{"thread_id": thread_id, "collection_name": thread_id} for  _ in chunks]
                                             )
        retriever = vector_store.as_retriever(
            search_type="similarity", search_kwargs={"k": 5}
        )

        _THREAD_RETRIEVERS[str(thread_id)] = retriever
        _THREAD_METADATA[str(thread_id)] = {
            "filename": filename or os.path.basename(temp_path),
            "documents": len(docs),
            "chunks": len(chunks),
        }

        return {
            "filename": filename or os.path.basename(temp_path),
            "documents": len(docs),
            "chunks": len(chunks),
        }
    finally:
        # The FAISS store keeps copies of the text, so the temp file is safe to remove.
        try:
            os.remove(temp_path)
        except OSError:
            pass


search_tool = DuckDuckGoSearchRun(region="us-en")


@tool
def create_file_tool(content:str, filename:str, mime_type:str)->dict:
    """Creates a downloadable file from content. Returns base64 bytes + metadata."""
    byte_data = content.encode("utf-8")
    return {
        "filename": filename,
         "mime_type":mime_type,
        "bytes_b64": base64.b64encode(byte_data).decode("utf-8")
    }

@tool
def rag_tool(query: str, thread_id: Optional[str] = None) -> dict:
    """
    Retrieve relevant information from the uploaded PDF for this chat thread.
    Always include the thread_id when calling this tool.
    """
    retriever = _get_retriever(thread_id)
    if retriever is None:
        return {
            "error": "No document indexed for this chat. Upload a PDF first.",
            "query": query,
        }

    result = retriever.invoke(query)
    context = [doc.page_content for doc in result]
    metadata = [doc.metadata for doc in result]

    return {
        "query": query,
        "context": context,
        "metadata": metadata,
        "source_file": _THREAD_METADATA.get(str(thread_id), {}).get("filename"),
    }

tools = [search_tool,rag_tool,create_file_tool]
tool_node = ToolNode(tools)



def chat_node(state: ChatState):
    # take user query from state
    query = state['messages']
    # send To LLM
    response = llm_with_tools.invoke(query)
    # response store state
    return {'messages':[response]};

#====================================== Graph ==========================================
llm_with_tools = model.bind_tools(tools)
graph = StateGraph(ChatState)
#add nodes
graph.add_node('chat_node', chat_node)
graph.add_node("tools", tool_node)

#add edges
graph.add_edge(START,'chat_node')
graph.add_conditional_edges("chat_node",tools_condition)
graph.add_edge('tools', 'chat_node')
# graph.add_edge('chat_node',END)

checkpointer = InMemorySaver()

chatbot = graph.compile(checkpointer=checkpointer)

# -------------------
# 8. Helpers
# -------------------
def retrieve_all_threads():
    all_threads = set()
    for checkpoint in checkpointer.list(None):
        all_threads.add(checkpoint.config["configurable"]["thread_id"])
    return list(all_threads)


def thread_has_document(thread_id: str) -> bool:
    return str(thread_id) in _THREAD_RETRIEVERS


def thread_document_metadata(thread_id: str) -> dict:
    return _THREAD_METADATA.get(str(thread_id), {})


CONFIG = {
        "configurable": {"thread_id": 1},
        "metadata": {"thread_id": 1},
        "run_name": "chat_turn",
    }
# while True:
#     user_message = input("\nType Here:")
#     initial_state = {
#         'messages':[HumanMessage(content=user_message)]
#     }
#     # config = {'configurable': {"thread_id" : thread_id}}
#     for update in chatbot.stream(initial_state,config=CONFIG,stream_mode="messages"):
#       print(update[0].content, end="", flush=True)
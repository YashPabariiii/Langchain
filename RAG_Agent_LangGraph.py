import os
import uuid
from typing import Annotated, List, TypedDict, Literal
from langchain_groq import ChatGroq
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langgraph.graph import StateGraph, END
from langchain_classic.indexes import SQLRecordManager, index
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from dotenv import load_dotenv

load_dotenv()

PDF_DATA_PATH = "./my_pdfs"      # Folder where you put your PDFs
DB_PATH = "./chroma_db_pdf"     # Folder where the VectorDB stays
RECORD_MANAGER_DB = "sqlite:///record_manager.sqlite"  # SQL DB for tracking indexed docs

if not os.path.exists(PDF_DATA_PATH):
    os.makedirs(PDF_DATA_PATH)
def get_incremental_retriever():
    """Syncs the folder with the VectorDB without deleting old data."""
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    # 1. Initialize the VectorStore
    vectorstore = Chroma(
        collection_name="pdf_collection",
        embedding_function=embeddings,
        persist_directory=DB_PATH
    )

    # 2. Setup the Record Manager (tracks what has been indexed)
    record_manager = SQLRecordManager(
        namespace="chroma/pdf_collection",
        db_url=RECORD_MANAGER_DB
    )
    record_manager.create_schema()

    # 3. Load and Split
    loader = PyPDFDirectoryLoader(PDF_DATA_PATH)
    raw_docs = loader.load()
    
    if not raw_docs:
        print("--- NO PDFs FOUND ---")
        return vectorstore.as_retriever()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=120,)
    chunks = text_splitter.split_documents(raw_docs)

    # 4. RUN INDEXING (The Magic Step)
    # cleanup="incremental" means: 
    # - Add new docs
    # - Skip unchanged docs
    # - Delete docs from the DB if they were removed from the folder
    indexing_stats = index(
        chunks,
        record_manager,
        vectorstore,
        cleanup="incremental",
        source_id_key="source",
        key_encoder="sha256"
    )

    print(f"--- INDEXING STATS: {indexing_stats} ---")
    # k=5: Retrieves the 5 most relevant segments for the LLM to analyze
    return vectorstore.as_retriever(search_kwargs={"k": 5})

retriever = get_incremental_retriever()

class RAGState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]
    context: List[str] # Injected knowledge snippets
    iterations: int

llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)

def retrieve_node(state: RAGState):
    """Retrieves relevant info from PDF VectorDB and stores it in context."""
    print("--- NODE: RETRIEVAL ---")
    query = state["messages"][-1].content
    if retriever is None:
        return {"context": ["No documents found in knowledge base."]}

    docs = retriever.invoke(query)
    return {"context": [d.page_content for d in docs]}

def grade_docs_edge(state: RAGState) -> Literal["generate", "rewrite"]:
    """Conditional Edge: Self-correct if retrieval is poor."""
    print("--- EDGE: GRADING RELEVANCE ---")
    if not state["context"] or len(state["context"][0]) < 20:
        return "rewrite"
    return "generate"

def rewrite_node(state: RAGState):
    """Rewrites the query for better PDF searching."""
    print("--- NODE: REWRITING QUERY ---")
    original = state["messages"][-1].content
    # Iteration check to prevent infinite loops
    if state.get("iterations", 0) > 3:
        return {"messages": [HumanMessage(content="Final attempt search...")]}

    msg = llm.invoke(f"Rewrite this question to be more specific for a PDF search: {original}")
    return {"messages": [HumanMessage(content=msg.content)], "iterations": state.get("iterations", 0) + 1}

def generate_node(state: RAGState):
    """Generates final answer using context."""
    print("--- NODE: GENERATE ANSWER ---")
    context_text = "\n".join(state["context"])
    user_query = state["messages"][0].content
    prompt = [
        SystemMessage(content=f"Use this PDF context to answer. If not found, say you don't know.\n\nContext:\n{context_text}"),
        HumanMessage(content=user_query)
    ]
    response = llm.invoke(prompt)
    return {"messages": [response]}


graph = StateGraph(RAGState)
graph.add_node("retrieve", retrieve_node)
graph.add_node("generate", generate_node)
graph.add_node("rewrite", rewrite_node)
graph.set_entry_point("retrieve")
graph.add_conditional_edges(
    "retrieve",
    grade_docs_edge,
    {
        "generate": "generate",
        "rewrite": "rewrite"
    }
)
graph.add_edge("rewrite", "retrieve")
graph.add_edge("generate",END)

checkpointer = MemorySaver()
app = graph.compile(checkpointer=checkpointer)

session_id = str(uuid.uuid4())
config_session = {"configurable": {"thread_id": session_id}}
print(f"\n[SYSTEM] Session Started with ID: {session_id}")

input_state = {
        "messages": [HumanMessage(content="How was NVIDIA's 2024 financial performance?")],
        "context": [],
        "iterations": 0
    }

for chunk in app.stream(input_state, config=config_session, stream_mode="values"):
    chunk["messages"][-1].pretty_print()

final_state = app.get_state(config_session)
print("\n" + "="*50)
print("FINAL PDF AGENT ANSWER:")
print(final_state.values["messages"][-1].content)
print("="*50)
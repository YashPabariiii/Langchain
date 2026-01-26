import uuid
import yfinance as yf
from typing import Annotated, List, TypedDict
from langchain_groq import ChatGroq
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.tools import tool
from langchain_core.messages import BaseMessage, HumanMessage , SystemMessage
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver
from dotenv import load_dotenv

load_dotenv()

# --- 1. DEFINE TOOLS ---
@tool
def get_stock_data(ticker: str):
    """Fetches real-time stock price and financial summary for a company ticker."""
    stock = yf.Ticker(ticker)
    info = stock.fast_info
    return {
            "ticker": ticker,
            "last_price": info['last_price'],
            "market_cap": info['market_cap'],
            "currency": info['currency']
        }

@tool
def web_search(query: str):
    """Searches the web for latest news, articles, and general information."""
    search = DuckDuckGoSearchRun()
    return search.run(query)

tools = [get_stock_data, web_search]
tool_node = ToolNode(tools)

class RAGAgentState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]
    documents: List[str] #Injected storage for retrieved documents
    iterations: int

llm= ChatGroq(model="llama-3.3-70b-versatile",temperature=0).bind_tools(tools)
search_tool = DuckDuckGoSearchRun()

def call_model(state: RAGAgentState):
    """The Brain: Decides whether to use a tool or not."""
    current_iter = state.get("iterations", 0)
    print(f"--- ITERATION {current_iter} --- to the model")

    sys_msg = SystemMessage(content=(
        "You are a tool-calling assistant. "
        "CRITICAL POINT: Output tool calls as RAW JSON ONLY. "
        "DO NOT use XML tags like <function> or <tool_call>. "
        "Example: {'name': 'get_stock_data', 'arguments': {'ticker': 'AAPL'}}"
    ))
    state["messages"] = [sys_msg] + state["messages"]
    response = llm.invoke(state["messages"])
    return {"messages": [response],"iterations": current_iter + 1}

def should_continue(state: RAGAgentState):
    """Routing Logic: Directs flow based on tool calls."""
    last_message = state["messages"][-1]
    if state.get("iterations", 0) > 5:
        print("--- MAX ITERATIONS REACHED ---")
        return END
    if last_message.tool_calls:
        return "tools"
    return END

graph = StateGraph(RAGAgentState)
graph.add_node("agent", call_model)
graph.add_node("tools", tool_node)
graph.set_entry_point("agent")
graph.add_conditional_edges(
    "agent",
    should_continue,{
        "tools": "tools",
        END: END
    }
)

# After the tools run, they MUST feed back the results into the agent for further reasoning.
graph.add_edge("tools", "agent")

# ---(INJECTED STATE) ---
# MemorySaver allows the agent to remember the thread even if the loop runs 10 times
checkpointer = MemorySaver()
app=graph.compile(checkpointer=checkpointer)

session_id = str(uuid.uuid4())
config_session_1={"configurable": {"thread_id": session_id}}
print(f"\n[SYSTEM] Session Started with ID: {session_id}")

query = {
    "messages": [
        HumanMessage(content="Find the ticker for NVIDIA, get its current price,and find one piece of recent news about their chips.")
    ],
    "documents": [],
    "iterations": 0
}

query2 = {
    "messages": [
        HumanMessage(content="Find the ticker for NVIDIA, get its current price,and find one piece of recent news about their chips.")],
    "documents": [],
    "iterations": 0
}

for chunk in app.stream(query, config=config_session_1,stream_mode="values"):
    chunk["messages"][-1].pretty_print()
    print(f"Agent: {chunk['messages'][-1].content}")

print("\n" + "="*60)
print("INTERNAL INJECTED STORAGE INSPECTION")
print("="*60)
final_state = app.get_state(config_session_1)
print(f"Final Checkpoint: {final_state.config['configurable']['checkpoint_id']}")
print(f"Messages in State: {(final_state.values['messages'])}")
print("="*60)
final_answer = final_state.values["messages"][-1].content
print("\n" + "="*30)
print("FINAL USER RESPONSE:")
print(final_answer)
print("="*30)
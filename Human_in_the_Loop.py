import os
from typing import Annotated, List, TypedDict
from langchain_groq import ChatGroq
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.tools import tool
from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver
from dotenv import load_dotenv

load_dotenv()

# --- 1. SETUP TOOLS ---
@tool
def terminal_notifier(message: str):
    """Sends a critical alert to the system admin terminal."""
    return f"ADMIN NOTIFIED: {message}"

search_tool = DuckDuckGoSearchRun()
tools = [search_tool, terminal_notifier]
tool_node = ToolNode(tools)

# --- 2. STATE DEFINITION ---
class AgentState(TypedDict):
    # 'add_messages' is the reducer that handles state updates
    messages: Annotated[List[BaseMessage], add_messages]

llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)
llm_with_tools = llm.bind_tools(tools)

def call_model(state: AgentState):
    return {"messages": [llm_with_tools.invoke(state["messages"])]}

def should_continue(state: AgentState):
    last_message = state["messages"][-1]
    if not last_message.tool_calls:
        return END
    return "tools"

# --- 4. GRAPH ASSEMBLY WITH INTERRUPT ---
graph = StateGraph(AgentState)
graph.add_node("agent", call_model)
graph.add_node("tools", tool_node)

graph.set_entry_point("agent")
graph.add_conditional_edges("agent", should_continue)
graph.add_edge("tools", "agent")

# Injected State: MemorySaver acts as the database for the agent's memory
memory = MemorySaver()

# HUMAN-IN-THE-LOOP: We pause the graph before 'tools' execute
app = graph.compile(checkpointer=memory, interrupt_before=["tools"])

# --- 5. EXECUTION LOOP (COLLABORATIVE) ---
config = {"configurable": {"thread_id": "revision_session_1"}}
query = {"messages": [HumanMessage(content="Search for the current weather in Surat and notify the admin.")]}

# STEP A: RUN UNTIL INTERRUPT
print("--- AGENT IS THINKING ---")
for chunk in app.stream(query, config, stream_mode="values"):
    chunk["messages"][-1].pretty_print()

# The graph is now paused because 'terminal_notifier' is a tool.
# We can inspect the state to see what it wants to do.
snapshot = app.get_state(config)
print(f"\n[PAUSED] Next Node: {snapshot.next}")
print(f"[PAUSED] Tool Call: {snapshot.values['messages'][-1].tool_calls[0]['name']}")

# STEP B: HUMAN INTERVENTION
user_choice = input("\nDo you want to allow this action? (y/n): ")

if user_choice.lower() == 'y':
    print("--- RESUMING EXECUTION ---")
    # Passing 'None' tells LangGraph to resume from the checkpoint
    for chunk in app.stream(None, config, stream_mode="values"):
        chunk["messages"][-1].pretty_print()
else:
    print("Action cancelled by human.")
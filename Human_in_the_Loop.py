import os
import uuid
from typing import Annotated, List, TypedDict
from langchain_groq import ChatGroq
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver
from dotenv import load_dotenv

load_dotenv()

# In the context of LangGraph, Injected State (or Injected Storage) refers to the ability to provide an external "memory" or "database" to the agent at runtime.
# Injected Storage is the "Memory Card." It allows the graph to:
# Checkpoint: Automatically save the state (messages, variables, variables) after every node execution.
# Resume: Use a unique ID (thread_id) to pull that specific "save file" back into the graph's working memory.
search_tool = DuckDuckGoSearchRun()
tools = [search_tool]
tool_node = ToolNode(tools)

class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]

llm = ChatGroq(model="llama-3.3-70b-versatile").bind_tools(tools)

def call_model(state: AgentState):
    return {"messages": [llm.invoke(state["messages"])]}

def route(state: AgentState):
    if state["messages"][-1].tool_calls:
        return "tools"
    return END

# --- 2. THE GRAPH WITH PERSISTENCE ---
graph = StateGraph(AgentState)
graph.add_node("agent", call_model)
graph.add_node("tools", tool_node)
graph.set_entry_point("agent")
graph.add_conditional_edges("agent", route)
graph.add_edge("tools", "agent")

# This is the "Injected Storage"
# In LangGraph, a checkpointer is the persistence layer that automatically saves the graph state at every super-step during execution. This enables features like human-in-the-loop, memory between runs, time travel, and fault tolerance.
checkpointer = MemorySaver()
app = graph.compile(checkpointer=checkpointer)

# --- 3. DEMONSTRATING AUTO SESSION ID ---

# Instead of hardcoding "ALICE", we generate a unique ID automatically.
# This is how real-world apps handle thousands of unknown users.
auto_id_1 = str(uuid.uuid4())
auto_id_2 = str(uuid.uuid4())

user_1_config = {"configurable": {"thread_id": auto_id_1}}
user_2_config = {"configurable": {"thread_id": auto_id_2}}

# Alice asks a question
print(f"--- USER 1 STEP 1 (ID: {auto_id_1}) ---")
# when we provide the thread_id The Graph doesn't start with a blank slate. It looks at the Injected Storage (Checkpointer) and asks: "Does 'user_42' have any saved history?" If yes, it loads all previous messages into the State.
app.invoke({"messages": [HumanMessage(content="Hi, my name is Alice.")]}, user_1_config)

# Bob asks a completely different question
print(f"--- USER 2 STEP 1 (ID: {auto_id_2}) ---")
app.invoke({"messages": [HumanMessage(content="What is the weather in Surat?")]}, user_2_config)

# NOW: We "Inject" Alice's state back into the graph using her auto-generated ID
print("\n--- USER 1 STEP 2 (Memory Injection) ---")
result = app.invoke({"messages": [HumanMessage(content="What is my name?")]}, user_1_config)
print(f"Agent to User 1: {result['messages'][-1].content}")

# AND: We check Bob's state using his auto-generated ID
print("\n--- USER 2 STEP 2 (Memory Injection) ---")
result = app.invoke({"messages": [HumanMessage(content="Which city did I ask about?")]}, user_2_config)
print(f"Agent to User 2: {result['messages'][-1].content}")

# --- 4. INTERNAL STATE INSPECTION (THE "HOW IT WORKS" PART) ---

# We use app.get_state() to see exactly what is currently 'injected' in the checkpointer
# This doesn't run the AI; it just reads the 'Memory Card' (Injected Storage)
print("\n" + "="*60)
print("INTERNAL WORKING: PEERING INTO THE INJECTED STORAGE")
print("="*60)

# Inspecting Alice's internal storage
# This shows how the graph 'hydrates' the state before processing
alice_internal = app.get_state(user_1_config)

print(f"[THREAD ID]: {user_1_config['configurable']['thread_id']}")
print(f"[CHECKPOINT ID]: {alice_internal.config['configurable']['checkpoint_id']}")
print(f"[NEXT NODE]: {alice_internal.next if alice_internal.next else 'End of Graph (Waiting for user)'}")
print("-" * 30)
print("RAW MESSAGES RETRIEVED FROM STORAGE:")

# This loop proves that the storage is holding the actual message objects
for i, msg in enumerate(alice_internal.values['messages']):
    role = "User" if isinstance(msg, HumanMessage) else "AI"
    print(f" {i+1}. [{role}]: {msg.content[:60]}...")

print("="*60)

# Final Confirmation: Sending one more message to see the ID update in real-time
print("\n[ACTION] Adding a new message to Alice's thread...")
app.invoke({"messages": [HumanMessage(content="I also like coding in Python.")]}, user_1_config)

# Re-fetching the state to show the version (Checkpoint ID) changed
new_alice_internal = app.get_state(user_1_config)
print(f"[UPDATE] New Checkpoint ID generated: {new_alice_internal.config['configurable']['checkpoint_id']}")
for i, msg in enumerate(new_alice_internal.values['messages']):
    role = "User" if isinstance(msg, HumanMessage) else "AI"
    print(f" {i+1}. [{role}]: {msg.content[:60]}...")
print("="*60)
import os
from typing import Annotated, List, TypedDict
from langchain_groq import ChatGroq
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.tools import tool
from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from IPython.display import display, Image
from dotenv import load_dotenv

load_dotenv()

# --- 1. DEFINE TOOLS ---

@tool
def analyze_sentiment(text: str):
    """
    Perform a deep sentiment analysis on a block of text.
    Useful for gauging public opinion on companies or news.
    """
    positive_words = ['growth', 'profit', 'success', 'up', 'buy']
    text_lower = text.lower()
    score = sum(1 for word in positive_words if word in text_lower)
    return "Positive" if score > 1 else "Neutral/Negative"

search_tool = DuckDuckGoSearchRun()

tools = [search_tool, analyze_sentiment]
tool_node = ToolNode(tools)

llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)
llm_with_tools = llm.bind_tools(tools)

# --- 2. DEFINE AGENT BEHAVIOR ---

class StockInfoAgent(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]

def call_model(state: StockInfoAgent):
    response = llm_with_tools.invoke(state['messages'])
    return {"messages": [response]}

def should_continue(state: StockInfoAgent):
    last_message = state['messages'][-1]

    if last_message.tool_calls:
        return "please continue"

    return END

# --- 3. BUILD THE AGENT GRAPH ---
graph = StateGraph(StockInfoAgent)

graph.add_node("tool_node", tool_node)
graph.add_node("call_model", call_model)
graph.add_node("should_continue", should_continue)

graph.set_entry_point("call_model")
graph.add_conditional_edges(
    "call_model",
    should_continue,
    {
        "please continue": "tool_node",
        END: END
    }
)
graph.add_edge("tool_node", "call_model")

app=graph.compile()

query = {
    "messages": [
        HumanMessage(content="Find recent news about NVIDIA's and Apple's stock performance. "
                             "Analyze the sentiment of the findings and summarize "
                             "if the outlook is generally positive.")
    ]
}

# --- OPTION 1: THE "BLACK BOX" APPROACH ---
# .invoke() runs the entire graph from start to finish.
# It returns only the FINAL state after the loop (END) is reached.
result = app.invoke(query)
print("-----------RESULT---------")
# This prints the dictionary containing the full message history
# after all tools have finished and the agent has summarized.
print(result)
print("-----------END------------")

# --- OPTION 2: THE "X-RAY" APPROACH ---
# .stream() allows you to watch the ReAct loop in real-time.
# Each 'chunk' is the state of the graph after a node (agent or tools) finishes.
for chunk in app.stream(query, stream_mode="values"):
    # We access [-1] because stream_mode="values" gives the whole list;
    # we only want to see the newest message produced in that step.
    chunk["messages"][-1].pretty_print()
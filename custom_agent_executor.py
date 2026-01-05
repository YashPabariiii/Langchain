from typing import List, Dict, Any
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage
from langchain_core.tools import tool
from langchain_core.runnables.base import RunnableSerializable
from dotenv import load_dotenv

load_dotenv()

# 2. DEFINE TOOLS
@tool
def get_stock_price(ticker: str) -> float:
    """Retrieves the current stock price for a given ticker symbol."""
    # Mock
    mock_data = {"AAPL": 175.50, "GOOGL": 140.20, "TSLA": 250.10}
    return mock_data.get(ticker.upper(), 100.0)

@tool
def get_company_news_sentiment(ticker: str) -> str:
    """Analyzes recent news for a company and returns a sentiment score (Positive/Negative/Neutral)."""
    # Mock analysis logic
    sentiments = {"AAPL": "Positive", "GOOGL": "Neutral", "TSLA": "Negative"}
    return sentiments.get(ticker.upper(), "Unknown")

@tool
def final_answer(answer: str, tools_used: List[str]) -> Dict[str, Any]:
    """Use this tool to provide the final structured response to the user once all research is done."""
    return {"answer": answer, "tools_used": tools_used}

tools = [get_stock_price, get_company_news_sentiment, final_answer]

#create tool name to function mapping
name2tool = {tool.name: tool.func for tool in tools}

# 3. DEFINE THE AGENT PROMPT

# MessagesPlaceholder is a special component used within a ChatPromptTemplate to reserve a specific spot for a list of messages that will be provided dynamically during the prompt's execution.
# chat_history: Holds previous Human and AI messages to give the agent long-term memory.
# agent_scratchpad: Holds the "internal transcript" of current tool calls (AI messages) and their results (Tool messages).

prompt = ChatPromptTemplate.from_messages([
    ("system", (
        "You are a Senior Financial Analyst. To answer user queries, you MUST use the provided tools. "
        "Combine price data and sentiment analysis to provide a recommendation. "
        "All intermediate steps go in the 'scratchpad'. Once you have the full analysis, "
        "use the 'final_answer' tool."
    )),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
])

# 4. CUSTOM AGENT EXECUTOR LOGIC
class CustomAgentExecutor:
    def __init__(self, model_name: str = "llama-3.3-70b-versatile", max_iterations: int = 3):
        # instance variable:hint what type it would be = initial value.
        self.chat_history:List[BaseMessage]=[]
        self.max_iterations=max_iterations
        llm=ChatGroq(model=model_name,temperature=0)
        self.agent: RunnableSerializable =(
            {
            # x is whatever you pass to .invoke().
            "input": lambda x:x["input"],
            "chat_history": lambda x:x["chat_history"],
            # We will use get() to make the agent safe on the first iteration, when no scratchpad exists yet.
            # else it will crash. this will intialise agent_scratchpad as an empty list if it doesn't exist.
            "agent_scratchpad": lambda x:x.get("agent_scratchpad",[])
            }
            | prompt
            | llm.bind_tools(tools,tool_choice="any")
        )

    def invoke(self,user_input:str):
        print(f"\n--- Starting Analysis for: {user_input} ---")
        iteration_count=0
        agent_scratchpad=[]
        final_llm_response= None

        while iteration_count<=self.max_iterations:
            llm_response=self.agent.invoke({
                "input": user_input,
                "chat_history": self.chat_history,
                "agent_scratchpad": agent_scratchpad
            })

            #  store the agent response in the sratchpad
            agent_scratchpad.append(llm_response)

            # retrieve the tools from the response
            tool_call=llm_response.tool_calls[0]
            tool_name=tool_call["name"]
            tool_args = tool_call["args"]
            tool_id = tool_call["id"]

            print(f"Iteration {iteration_count}: Calling tool '{tool_name}' with {tool_args} and {tool_id}")

            tool_response=name2tool[tool_name](**tool_args)
            agent_scratchpad.append(ToolMessage(content=str(tool_response),tool_call_id=tool_id))

            iteration_count+=1

            if tool_name=="final_answer":
                final_llm_response=tool_response
                break

        if final_llm_response:
            self.chat_history.extend([HumanMessage(content=user_input),AIMessage(content=str(final_llm_response))])

        return final_llm_response           


if __name__ == "__main__":
    executor = CustomAgentExecutor()
    
    # Example: Requires calling get_stock_price, then get_company_news_sentiment, then final_answer
    result = executor.invoke("Is it a good time to buy Google?")
    
    print("\n--- FINAL STRUCTURED RESULT ---")
    print(f"\n Answer: {result['answer']}")
    print(f"\n Tools Used: {result['tools_used']}")                     
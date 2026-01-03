import requests
from typing import Dict

from langchain_groq import ChatGroq
from langchain_core.tools import tool
# create_tool_calling_agent → creates an LLM that can reason + call tools
#AgentExecutor → runs the reasoning loop
from langchain.agents import AgentExecutor, create_tool_calling_agent

from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv

load_dotenv()

llm = ChatGroq(model_name="llama-3.3-70b-versatile",temperature=0)

@tool
def get_city_aqi(city: str) -> Dict[str, int]:
    """
    Fetch current AQI for a city using WAQI API.
    """
    url = f"https://api.waqi.info/feed/{city}/?token=demo"
    response = requests.get(url, timeout=10)
    response.raise_for_status()

    data = response.json()
    if data["status"] != "ok":
        raise ValueError("AQI data unavailable")

    return {"aqi": int(data["data"]["aqi"])}

@tool
def outdoor_exercise_advice(aqi: int) -> Dict[str, str]:
    """
    Give outdoor exercise advice based on AQI.
    """
    if aqi <= 50:
        return {"advice": "Air quality is good. Exercise outdoors freely."}
    elif aqi <= 100:
        return {"advice": "Moderate air quality. Sensitive people should limit exercise."}
    elif aqi <= 150:
        return {"advice": "Unhealthy for sensitive groups. Avoid heavy workouts."}
    else:
        return {"advice": "Poor air quality. Avoid outdoor exercise."}

tools = [get_city_aqi, outdoor_exercise_advice]

#ChatPromptTemplate.from_messages is where you define everything about how the agent talks, thinks, and uses tools.
# Inside this we can give all the prompts system,human,etc.

prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You are a helpful assistant that can call tools to answer questions. "
     "Use tools when needed and combine their outputs logically."),
    ("human", "{input}"),
    #This is where:Past thoughts, Tool calls , Observations are stored between steps. Without this → agent breaks.
    ("placeholder", "{agent_scratchpad}")
])

# CREATE AGENT
agent = create_tool_calling_agent(llm=llm,tools=tools,prompt=prompt)

agent_executor = AgentExecutor(agent=agent,tools=tools,
    verbose=True   # IMPORTANT: shows reasoning + tool calls
    )

# RUN AGENT
response = agent_executor.invoke({"input": "What is the current AQI in Chicago and should I exercise outdoors?"})

print("\n===== FINAL AGENT OUTPUT =====")
print(response["output"])

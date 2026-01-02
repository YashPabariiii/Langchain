import requests
import json
from typing import Dict
from langchain_groq import ChatGroq
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, ToolMessage
from dotenv import load_dotenv

load_dotenv()

llm = ChatGroq(model_name="llama-3.3-70b-versatile",temperature=0)

# -----------------------------
# Tool 1: Fetch temperature
# -----------------------------
@tool
def get_temperature_celsius(city: str) -> Dict[str, float]:
    """
    Get current temperature in Celsius for a city.
    """
    response = requests.get(f"https://wttr.in/{city}?format=j1",timeout=100)
    response.raise_for_status()

    data = response.json()
    temp_c = float(data["current_condition"][0]["temp_C"])

    return {"temperature_celsius": temp_c}


# -----------------------------
# Tool 2: Clothing advice
# -----------------------------
@tool
def clothing_advice(temperature_celsius: float) -> Dict[str, str]:
    """
    Give clothing advice based on temperature.
    """
    if temperature_celsius >= 30:
        advice = "Wear light cotton clothes and stay hydrated."
    elif temperature_celsius >= 20:
        advice = "Comfortable casual wear is fine."
    else:
        advice = "Wear warm clothes or a light jacket."

    return {"advice": advice}

llm_with_tools=llm.bind_tools([get_temperature_celsius,clothing_advice])

messages= [HumanMessage(content="What is the current temperature in Mumbai, and based on that what should I wear?")]

first_llm_response=llm_with_tools.invoke(messages)
print("\n===== LLM RESPONSE 1 =====")
print(first_llm_response)

messages.append(first_llm_response)

print("\n===== TOOL CALLS 1 =====")
tool_call_1 = first_llm_response.tool_calls[0]
print(tool_call_1)

#invoke() expects ONLY args, not the whole tool_call object.
tool_result_1 = get_temperature_celsius.invoke(tool_call_1["args"])
print("\n===== TOOL 1 RESPONSE =====")
print(tool_result_1)

# Tool output MUST be wrapped in ToolMessage and MUST include tool_call_id
messages.append(ToolMessage(content=json.dumps(tool_result_1),tool_call_id=tool_call_1['id']))

# post the use of tool suggested by LLM we will call the LLM again with tool output
second_llm_response=llm_with_tools.invoke(messages)
print(second_llm_response)

print("\n===== TOOL CALLS 2 =====")
print(second_llm_response.tool_calls)
messages.append(second_llm_response)

# here we got the advice to use second tool so boiiii le's do that
tool_call_2 = second_llm_response.tool_calls[0]
print("\n===== LLM RESPONSE 2 =====")
print(second_llm_response)

# Now as we know tool 2 needs the input from output of tool 1 we will append that.

# DEPENDENCY INJECTION
tool_call_2["args"]["temperature_celsius"] = tool_result_1["temperature_celsius"]

tool_result_2=clothing_advice.invoke(tool_call_2['args'])
print("\n===== TOOL 2 RESPONSE =====")
print(tool_result_2)
messages.append(ToolMessage(content=json.dumps(tool_result_2),tool_call_id=tool_call_2['id']))

final_response = llm_with_tools.invoke(messages)

print("\n===== FINAL ANSWER =====")
print(final_response.content)
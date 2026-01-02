import os
from typing import Dict
from langchain_groq import ChatGroq
from langchain_core.messages.human import HumanMessage
from langchain_core.tools import tool
from pypdf import PdfReader
from dotenv import load_dotenv

load_dotenv()
llm = ChatGroq(model_name="llama-3.3-70b-versatile",temperature=0)

@tool
def count_words_in_pdf(file_path: str) -> Dict[str, int]:
    """
    Count total words in a PDF file.

    Args:
        file_path: Absolute or relative path to the PDF file

    Returns:
        Total word count
    """
    if not os.path.exists(file_path):
        raise ValueError("PDF file does not exist")

    reader = PdfReader(file_path)
    full_text = ""

    for page in reader.pages:
        text = page.extract_text()
        if text:
            full_text += text + " "

    word_count = len(full_text.split())

    return {
        "word_count": word_count
    }

# bind_tools expects a list
llm_with_tools=llm.bind_tools([count_words_in_pdf]) 

pdf_path = "/Users/limbanijayhasmukhbhai/Yash/EJ1172284.pdf"

messages= [HumanMessage(content=f"Count the number of words in the pdf file: {pdf_path}")]

response=llm_with_tools.invoke(messages)

print("\n===== RAW LLM RESPONSE =====")
print(response)

print("\n===== TOOL CALLS =====")
print(response.tool_calls)

messages.append(response)

#Now the LLM has given us idea that we should use the tool and args for that now we will implement the tool.
#Hence we will invoke our tool , here the args will be the first tool call suggested by the LLM.
tool_result=count_words_in_pdf.invoke(response.tool_calls[0])
print("\n===== TOOL RESPONSE =====")
print(tool_result)

# We will now append our tool result to the messages
messages.append(tool_result)

final_result=llm_with_tools.invoke(messages)

print("\n===== FINAL LLM RESPONSE =====")
print(final_result.content)
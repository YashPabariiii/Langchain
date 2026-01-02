from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_core.messages.human import HumanMessage
from dotenv import load_dotenv

load_dotenv()

prompt = ChatPromptTemplate.from_template("tell me a short joke about {topic}")
model = ChatGroq(model_name="llama-3.3-70b-versatile")
output_parser = StrOutputParser()

chain = prompt | model | output_parser

result=chain.invoke({"topic": "ice cream"})
# print(result)
# print(prompt.invoke({"topic": "ice cream"}))

messages = [HumanMessage(content='tell me a short joke about ice cream')]
print(model.invoke(messages))
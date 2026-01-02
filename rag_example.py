from langchain_groq import ChatGroq
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from dotenv import load_dotenv

load_dotenv()

# --- 1. SETUP THE KNOWLEDGE BASE ---
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

texts = [
    "The Alpha Centauri system is the closest star system to the Solar System.",
    "Cats are obligate carnivores and love eating tuna.",
    "The Eiffel Tower was completed on March 31, 1889."
]

vectorstore = FAISS.from_texts(texts, embedding=embeddings)
retriever = vectorstore.as_retriever()

# --- 2. DEFINE THE TEMPLATE ---
template = """Answer the question based only on the following context:
{context}

Question: {question}
"""
prompt = ChatPromptTemplate.from_template(template=template)

# --- 3. HELPER FUNCTION ---
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# --- 4. THE GROQ RAG CHAIN ---
llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0
)

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# --- 5. EXECUTION ---
query = "What is the diet of a cat according to the text?"

print(f"--- Querying Groq RAG Chain ---")
print(f"Question: {query}")

# Running the chain.
response = rag_chain.invoke(query)

print(f"\n--- Groq AI Response ---")
print(response)

# --- BONUS: DEBUGGING THE PIPELINE ---
print(f"\n--- Detailed Context Check ---")
context_docs = retriever.invoke(query)
print(f"Retriever pulled {len(context_docs)} relevant documents.")
for i, doc in enumerate(context_docs):
    print(f"Doc {i+1}: {doc.page_content}")
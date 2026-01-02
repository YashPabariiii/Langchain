# rag_example.py

## Core Components

* **HuggingFaceEmbeddings**: Converts your text sentences into lists of numbers (vectors). This allows the computer to understand that "tuna" and "fish" are mathematically similar.
* **FAISS**: A vector database. It stores these numbers and allows for lightning-fast searching.
* **Retriever**: Think of this as a Librarian. You give it a keyword, and it goes into the FAISS database to find the most relevant "pages" (documents).

---

## The RAG Chain (The Assembly Line)

This is the heart of **LCEL (LangChain Expression Language)**. It uses the `|` (pipe) operator to move data through the following steps:

### 1. Input Dictionary `{"context": ..., "question": ...}`

* **Context Branch**: Takes your query  runs it through the **retriever**  cleans it with `format_docs`.
* **Question Branch**: Uses `RunnablePassthrough()` to take the query and pass it forward unchanged.

### 2. Prompt

Takes that dictionary and plugs the values into the `{context}` and `{question}` slots in your template.

### 3. LLM (Groq)

Receives the formatted prompt and generates a response.

### 4. StrOutputParser

The LLM actually returns a complex object containing timestamps and token counts; this parser strips all that away and gives you just the text of the answer.

from langchain_core.runnables import (
    RunnablePassthrough, 
    RunnableLambda, 
    RunnableParallel
)

# --- HELPER FUNCTIONS ---

def get_doc_summary(data: dict):
    """Simulates extracting a key and transforming it."""
    content = data.get("content", "empty")
    return f"SUMMARY: {content[:20]}..."

def generate_priority_score(input_data):
    """Simulates generating a numeric value."""
    # Just a mock score based on content length
    return len(str(input_data)) * 10

def metadata_extractor(input_dict: dict):
    """Extracts a specific key."""
    return input_dict.get("priority_score", "0")

def format_final_output(value):
    """Final string transformation."""
    return f"FINAL PROCESSED STATUS: {str(value).upper()}"

# --- BUILDING THE CHAINS ---

# 1. Setup the Parallel Processing branch
print("--- Step 1: Parallel Branching ---")
branch_chain = RunnableParallel({
    "metadata": RunnablePassthrough() | RunnableLambda(get_doc_summary),
    "raw_copy": lambda z: z["content"]
})
print(branch_chain)

# 2. Setup the Assign Chain
# The .assign() method is powerful: it keeps existing keys and adds a new one.
print("--- Step 2: Using .assign() to append data ---")
processing_chain = branch_chain.assign(
    priority_score=RunnableLambda(generate_priority_score)
)
print(processing_chain)

# 3. Setup the Extraction Chain
print("--- Step 3: Formatting Pipeline ---")
output_formatting_chain = (
    RunnableLambda(metadata_extractor) | 
    RunnableLambda(format_final_output)
)
print(output_formatting_chain)

# 4. The Final Combined Chain
print("--- Step 4: Full Analysis Pipeline ---")
full_analysis_pipeline = processing_chain | output_formatting_chain
print(full_analysis_pipeline)

# --- EXECUTION ---

doc_input = {
    "content": "Urgent: The server is overheating in the data center.",
    "author": "Admin"
}

print("\n--- RUNNING THE COMPLETE PIPELINE ---")

initial_state=branch_chain.invoke(doc_input)
print(f"Initial Data (Before Assign):\n{initial_state}\n")

intermediate_state = processing_chain.invoke(doc_input)
print(f"Intermediate Data (After Assign):\n{intermediate_state}\n")

final_result = full_analysis_pipeline.invoke(doc_input)
print(f"Final Result:\n{final_result}")
from langchain_core.runnables import RunnablePassthrough, RunnableLambda, RunnableParallel

# --- SETUP HELPER FUNCTIONS ---

def extract_sentiment(text: str):
    """A simple 'complex' logic to simulate sentiment analysis."""
    text = text.lower()
    if any(word in text for word in ["bad", "broken", "angry", "slow"]):
        return "HIGH PRIORITY"
    return "STANDARD"

def format_log(data: dict):
    """Simulates formatting data for a database or UI."""
    return f"Ticket [{data['id']}]: {data['content']} -> Status: {data['status']}"

# --- 1. THE PASS-THROUGH CHAIN ---
# Purpose: Shows how data flows unchanged through multiple steps.
print("--- 1. Simple Passthrough Chain ---")
simple_chain = RunnablePassthrough() | RunnablePassthrough()
result1 = simple_chain.invoke("Ticket-101")
print(f"Result: {result1}\n")


# --- 2. THE LAMBDA TRANSFORMATION ---
# Purpose: Uses RunnableLambda to modify the data in the middle of the pipe.
print("--- 2. Lambda Transformation ---")
lambda_chain = RunnablePassthrough() | RunnableLambda(extract_sentiment)
result2 = lambda_chain.invoke("The internet connection is very slow and broken")
print(f"Sentiment Analysis Result: {result2}\n")


# --- 3. THE PARALLEL PROCESSOR ---
# Purpose: Branching the data to perform multiple operations at once.
# 'x' keeps the original data, 'y' extracts specific info, 'z' runs logic.
print("--- 3. Complex Parallel Processing ---")
parallel_chain = RunnableParallel({
    "original_input": RunnablePassthrough(),
    "id_only": lambda x: x["id"],
    "priority": lambda x: extract_sentiment(x["content"])
})

input_data = {
    "id": "TKT-404",
    "content": "My screen is flickering and I am angry"
}

result3 = parallel_chain.invoke(input_data)
print("Parallel Output Dictionary:")
for key, value in result3.items():
    print(f"  {key}: {value}")


# --- 4. COMBINING IT ALL ---
# A realistic pipeline: Input -> Parallel Processing -> Final Formatting
print("\n--- 4. Full Pipeline ---")
full_pipeline = (
    RunnableParallel({
        "id": lambda x: x["id"],
        "content": lambda x: x["content"].upper(),
        "status": lambda x: extract_sentiment(x["content"])
    }) 
    | RunnableLambda(format_log)
)

final_output = full_pipeline.invoke({
    "id": "TKT-999", 
    "content": "Everything is working perfectly fine!"
})
print(f"Final Formatted Log:\n{final_output}")
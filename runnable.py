from abc import ABC, abstractmethod

# 1. The Base Runnable Class
class CRunnable(ABC):
    def __init__(self):
        self.next = None

    @abstractmethod
    def process(self, data):
        pass

    def invoke(self, data):
        processed_data = self.process(data)
        if self.next is not None:
            return self.next.invoke(processed_data)
        return processed_data

    def __or__(self, other):
        # This allows: component1 | component2
        return CRunnableSequence(self, other)

# 2. The Sequence Class (The "Glue")
class CRunnableSequence(CRunnable):
    def __init__(self, first, second):
        super().__init__()
        self.first = first
        self.second = second

    def process(self, data):
        return data

    def invoke(self, data):
        # Step-by-step handoff
        first_result = self.first.invoke(data)
        return self.second.invoke(first_result)

# --- Implementation Examples with Logging ---

class SimplePrompt(CRunnable):
    def process(self, data):
        print(f"[STEP 1: PROMPT] Formatting input: {data}")
        return f"Who is {data}"

class SimpleModel(CRunnable):
    def process(self, data):
        print(f"[STEP 2: MODEL] Receiving prompt: '{data}'")
        # Simulating an AI response
        return "AI Response: He is a cricketer"

class SimpleParser(CRunnable):
    def process(self, data):
        print(f"[STEP 3: PARSER] Cleaning up the model output...")
        # Simulating extracting just the text (removing the 'AI Response:' prefix)
        return data.replace("AI Response: ", "")

# --- Running the Chain ---

# Initialize components
prompt = SimplePrompt()
model = SimpleModel()
parser = SimpleParser()

# Create the chain using the pipe operator
# Logic: (SimplePrompt | SimpleModel) -> creates a sequence, then | SimpleParser -> wraps it again
chain = prompt | model | parser

print("--- Starting Chain Execution ---")
result = chain.invoke("Virat Kohli")
print("--- Chain Execution Finished ---")

print(f"\nFINAL OUTPUT: {result}")
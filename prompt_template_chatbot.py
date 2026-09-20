
import os

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate

# Load environment variables
load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY not found in .env file")

# Initialize Gemini
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=GOOGLE_API_KEY,
    temperature=0.7
)

# Create prompt template
prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """
        You are a helpful AI assistant.
        Explain concepts in a simple and clear manner.
        Provide examples whenever possible.
        """
    ),
    (
        "human",
        "{user_input}"
    )
])

# Create chain
chain = prompt | llm


# Chat function
def chat_with_gemini(user_input):

    try:

        response = chain.invoke({
            "user_input": user_input
        })

        return response.content

    except Exception as e:

        return f"Error: {e}"


# Run chatbot
def run_chatbot():

    print("=" * 50)
    print("       ADVANCED LANGCHAIN CHATBOT")
    print("=" * 50)
    print("Type 'exit' or 'quit' to stop.")

    while True:

        user_input = input("\nYou: ").strip()

        if user_input.lower() in ["exit", "quit"]:
            print("Goodbye!")
            break

        if not user_input:
            continue

        print("Gemini is thinking...")

        response = chat_with_gemini(user_input)

        print("\nGemini:", response)


if __name__ == "__main__":
    run_chatbot()
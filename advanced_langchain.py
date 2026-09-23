
import os

from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI

from langchain_core.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder
)

from langchain_core.messages import (
    HumanMessage,
    AIMessage
)


# ==========================================
# 1. Load Environment Variables
# ==========================================

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    raise ValueError(
        "GOOGLE_API_KEY not found in .env file"
    )


# ==========================================
# 2. Initialize Gemini Model
# ==========================================

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=GOOGLE_API_KEY,
    temperature=0.7
)


# ==========================================
# 3. Create Advanced Prompt Template
# ==========================================

prompt = ChatPromptTemplate.from_messages([

    (
        "system",
        """
You are an expert AI Learning Assistant.

Your goal is to help the user learn
Machine Learning, Deep Learning, LLMs,
Python, Data Science, and Computer Science.

IMPORTANT INSTRUCTIONS:

1. If the user provides a single word
   or short topic, explain it in detail.

2. If the user asks a question,
   answer clearly and accurately.

3. If the user provides a custom prompt,
   follow the user's instructions.

4. Use simple language for beginners.

5. Provide real-world examples.

6. Include mathematical intuition
   when relevant.

7. Provide Python code when requested
   or when it helps understanding.

8. Explain complex topics step by step.

9. Use headings, bullet points, and
   examples for readability.

10. If the topic is ambiguous,
    ask for clarification when necessary.

11. Use previous conversation context
    to answer follow-up questions.

Do not assume every input is a topic.
Respect the user's actual instructions.
"""
    ),

    MessagesPlaceholder(
        variable_name="history"
    ),

    (
        "human",
        "{user_input}"
    )

])


# ==========================================
# 4. Create Chain
# ==========================================

chain = prompt | llm


# ==========================================
# 5. Conversation Memory
# ==========================================

chat_history = []


# ==========================================
# 6. Chat Function
# ==========================================

def chat_with_gemini(user_input):

    try:

        response = chain.invoke({

            "history": chat_history,

            "user_input": user_input

        })

        # Store user message
        chat_history.append(
            HumanMessage(content=user_input)
        )

        # Store AI response
        chat_history.append(
            AIMessage(content=response.content)
        )

        return response.content

    except Exception as e:

        return f"Error: {e}"


# ==========================================
# 7. Run Chatbot
# ==========================================

def run_chatbot():

    print("=" * 60)
    print("       FAISAL AI LEARNING ASSISTANT")
    print("=" * 60)

    print("Ask any question or enter a topic.")
    print("Type 'clear' to clear memory.")
    print("Type 'exit' or 'quit' to stop.")

    print("=" * 60)

    while True:

        user_input = input("\nYou: ").strip()

        if user_input.lower() in ["exit", "quit"]:

            print("Goodbye!")
            break

        if user_input.lower() == "clear":

            chat_history.clear()

            print("Conversation memory cleared!")

            continue

        if not user_input:

            print("Please enter a message.")

            continue

        print("\nGemini is thinking...")

        response = chat_with_gemini(user_input)

        print("\nGemini:\n")
        print(response)


# ==========================================
# 8. Main Entry Point
# ==========================================

if __name__ == "__main__":

    run_chatbot()
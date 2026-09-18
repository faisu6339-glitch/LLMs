
import os
import google.generativeai as genai
from dotenv import load_dotenv

# ==========================================
# 1. Load Environment Variables
# ==========================================

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    raise ValueError(
        "GOOGLE_API_KEY not found in .env file"
    )

print("API key loaded successfully!")


# ==========================================
# 2. Configure Gemini API
# ==========================================

genai.configure(api_key=GOOGLE_API_KEY)


# ==========================================
# 3. Initialize Model
# ==========================================

model = genai.GenerativeModel(
    "gemini-2.5-flash"
)


# ==========================================
# 4. Chat Function
# ==========================================

def chat_with_gemini(user_input):

    try:

        response = model.generate_content(
            user_input
        )

        return response.text

    except Exception as e:

        return f"Error: {e}"


# ==========================================
# 5. Run Chatbot
# ==========================================

def run_chatbot():

    print("=" * 50)
    print("       GEMINI AI CHATBOT")
    print("=" * 50)
    print("Type 'exit' or 'quit' to stop.")
    print("=" * 50)

    while True:

        user_input = input("\nYou: ").strip()

        if not user_input:
            print("Please enter a message.")
            continue

        if user_input.lower() in ["exit", "quit"]:
            print("Exiting chatbot. Goodbye!")
            break

        print("Gemini is thinking...")

        response = chat_with_gemini(user_input)

        print("\nGemini:", response)


# ==========================================
# 6. Main Entry Point
# ==========================================

if __name__ == "__main__":

    run_chatbot()
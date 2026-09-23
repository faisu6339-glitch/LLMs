from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from dotenv import load_dotenv
import os


# ============================================================
# LOAD ENVIRONMENT VARIABLES
# ============================================================

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    raise ValueError(
        "GOOGLE_API_KEY not found in .env file"
    )


# ============================================================
# INITIALIZE GEMINI
# ============================================================

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=GOOGLE_API_KEY,
    temperature=0.4,
    max_retries=5,
    timeout=60
)


# ============================================================
# LAPTOP PROMPT
# ============================================================

prompt = ChatPromptTemplate.from_messages([

    (
        "system",
        """
You are an expert laptop buying assistant.

Your job is to analyze the user's requirements and
help them understand what type of laptop they need.

Do NOT blindly recommend an expensive laptop.

Analyze the requirements carefully and explain:

1. Recommended processor
2. Recommended RAM
3. Recommended storage
4. Recommended GPU
5. Recommended display
6. Recommended battery capacity
7. Recommended cooling system
8. Recommended operating system
9. Upgradeability
10. Ports
11. Build quality
12. Weight and portability
13. Webcam
14. Keyboard
15. Speakers
16. Wi-Fi and Bluetooth
17. Warranty considerations

Also explain WHY each specification matters.

Compare the requirements against:

- Programming
- Machine Learning
- Deep Learning
- Data Science
- Android Studio
- VS Code
- Gaming
- Video editing

If the user has not provided a requirement,
do not invent a personal preference.

Clearly distinguish:
- Minimum requirement
- Recommended requirement
- Ideal requirement

Do not claim current prices or current availability
unless actual current product data is provided.
"""
    ),

    (
        "user",
        """
I want to buy a laptop.

Here are my requirements:

Budget:
{budget}

Main purpose:
{purpose}

Programming languages:
{programming_languages}

Software/tools:
{software_tools}

Machine Learning:
{machine_learning}

Deep Learning:
{deep_learning}

Gaming:
{gaming}

Video editing:
{video_editing}

RAM requirement:
{ram}

Storage requirement:
{storage}

GPU requirement:
{gpu}

Display requirement:
{display}

Battery requirement:
{battery}

Preferred brands:
{brands}

Operating system:
{operating_system}

Portability:
{portability}

Other requirements:
{other_requirements}


Now analyze my requirements and provide a complete
laptop buying guide.

Include:

1. My requirement summary

2. Minimum specifications I should buy

3. Recommended specifications

4. Ideal specifications

5. CPU explanation

6. RAM explanation

7. SSD explanation

8. GPU explanation

9. Display explanation

10. Battery explanation

11. Cooling explanation

12. Ports and connectivity

13. Upgradeability

14. What specifications I should NOT compromise on

15. What specifications I can compromise on

16. What type of laptop I should search for

17. Suitable laptop categories

18. Important things to check before purchasing

19. Common mistakes to avoid

20. Final buying checklist
"""
    )
])


# ============================================================
# CREATE CHAIN
# ============================================================

chain = prompt | llm | StrOutputParser()


# ============================================================
# USER REQUIREMENTS
# ============================================================

requirements = {

    "budget": "₹80,000",

    "purpose":
        "Programming, Machine Learning, Deep Learning, "
        "Data Science, Android development and gaming",

    "programming_languages":
        "Python, Java, JavaScript, C++",

    "software_tools":
        "VS Code, Android Studio, PyCharm, "
        "Jupyter Notebook, Google Colab",

    "machine_learning":
        "Yes, model training and experimentation",

    "deep_learning":
        "Yes, CNN, TensorFlow, PyTorch and LLM learning",

    "gaming":
        "Yes, modern games at 1080p",

    "video_editing":
        "Occasional video editing",

    "ram":
        "At least 16GB, preferably upgradeable to 32GB",

    "storage":
        "At least 512GB SSD, preferably 1TB",

    "gpu":
        "Dedicated NVIDIA GPU preferred",

    "display":
        "15 or 16 inch, Full HD or better, good colors",

    "battery":
        "Good battery life for programming and college",

    "brands":
        "Lenovo, ASUS, HP, Dell, Acer",

    "operating_system":
        "Windows 11",

    "portability":
        "Moderate; performance is more important than ultra-light weight",

    "other_requirements":
        """
Good cooling,
comfortable keyboard,
Wi-Fi 6 or better,
Bluetooth,
USB Type-C,
HDMI,
good build quality,
and reliable after-sales service.
"""
}


# ============================================================
# RUN CHAIN
# ============================================================

try:

    print("\n")
    print("=" * 70)
    print("              AI LAPTOP BUYING ASSISTANT")
    print("=" * 70)

    print("\nGemini is analyzing your requirements...\n")

    response = chain.invoke(requirements)

    print("=" * 70)
    print("                    LAPTOP ANALYSIS")
    print("=" * 70)

    print(response)

    print("\n")
    print("=" * 70)
    print("                  END OF ANALYSIS")
    print("=" * 70)


except Exception as e:

    print("\n❌ Error while contacting Gemini")

    print("\nError:")
    print(e)

    print("\nTry running the program again if Gemini")
    print("temporarily returns a 503 error.")
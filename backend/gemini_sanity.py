# backend/gemini_sanity.py

import os
from dotenv import load_dotenv
import google.generativeai as genai

def main():
    load_dotenv()

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is not set in .env")

    genai.configure(api_key=api_key)

    # Use a universally valid model name
    model = genai.GenerativeModel("models/gemini-2.5-flash")

    prompt = "Say one short sentence confirming that Gemini API works."

    try:
        response = model.generate_content(prompt)
    except Exception as e:
        print("❌ Gemini call failed:")
        print(repr(e))
        return

    text = getattr(response, "text", None)
    if not text and hasattr(response, "candidates") and response.candidates:
        text = response.candidates[0].content.parts[0].text

    print(f"✅ Gemini response: ")
    print(text)

if __name__ == "__main__":
    main()

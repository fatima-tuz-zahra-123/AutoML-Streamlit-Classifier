import google.generativeai as genai
import os

# The key from secrets.toml
api_key = "AIzaSyC_h6bO9Sr2ydwIfRN3wkIQ1ZH9_DEVOIc"

genai.configure(api_key=api_key)

print("Checking available models...")
try:
    for m in genai.list_models():
        if 'generateContent' in m.supported_generation_methods:
            print(f"- {m.name}")
except Exception as e:
    print(f"Error listing models: {e}")

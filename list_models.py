import google.generativeai as genai
import os
import streamlit as st

# Try to get key from secrets or environment
api_key = "AIzaSyC_h6bO9Sr2ydwIfRN3wkIQ1ZH9_DEVOIc"

try:
    genai.configure(api_key=api_key)
    print("Listing available models:")
    for m in genai.list_models():
        if 'generateContent' in m.supported_generation_methods:
            print(m.name)
except Exception as e:
    print(f"Error: {e}")

import google.generativeai as genai
import os

genai.configure(api_key="nkjjb")

for m in genai.list_models():
    print(m.name)

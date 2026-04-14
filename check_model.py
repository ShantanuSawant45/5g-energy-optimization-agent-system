import google.generativeai as genai
import os

genai.configure(api_key="AIzaSyC-Lqy0jgU-rKnf-bBv2FtfQeikesj7ch4")

for m in genai.list_models():
    print(m.name)
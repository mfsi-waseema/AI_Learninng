import streamlit as st
import requests
import os
from dotenv import load_dotenv

load_dotenv()
API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")

st.title("Smart Research Assistant Demo")

query = st.text_input("Ask something")

if query:
    response = requests.get(
        f"{API_BASE_URL}/ask",
        params={"q": query},
        stream=True,
        timeout=60,
    )
    placeholder = st.empty()
    output = ""
    if response.ok:
        for chunk in response.iter_content(chunk_size=32):
            if chunk:
                output += chunk.decode()
                placeholder.markdown(output)
    else:
        st.error(f"API request failed: {response.status_code}")
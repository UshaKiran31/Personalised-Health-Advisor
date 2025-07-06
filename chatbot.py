# chatbot/chatbot.py
import requests
import streamlit as st

API_URL = "https://api.groq.com/openai/v1/chat/completions"

def get_headers():
    """Get headers with API key, handling the case when it's not available."""
    try:
        return {
            "Authorization": f"Bearer {st.secrets['groq_api_key']}",
            "Content-Type": "application/json"
        }
    except KeyError:
        # During local development without secrets
        st.warning("Groq API key not found. Chatbot functionality will be limited.")
        return {
            "Content-Type": "application/json"
        }

def generate_response(user_message):
    """Generate a response to the user's message using the Groq API."""
    headers = get_headers()
    
    # Check if API key is available
    if "Authorization" not in headers:
        return "Sorry, I can't process your request right now. The API key is not configured."
    
    payload = {
        "model": "llama3-8b-8192",  # You can change this to llama3-70b if needed
        "messages": [
            {"role": "system", "content": "You are a helpful Health assistant. Answer questions about their Personalised Health and Diet Plan."},
            {"role": "user", "content": user_message}
        ],
        "temperature": 0.7
    }

    try:
        response = requests.post(API_URL, headers=headers, json=payload)
        response.raise_for_status()  # Raise an exception for HTTP errors
        result = response.json()
        return result["choices"][0]["message"]["content"]
    except requests.exceptions.RequestException as e:
        return f"Sorry, there was an error communicating with the API: {str(e)}"
    except (KeyError, IndexError) as e:
        return f"Sorry, there was an error processing the response: {str(e)}"
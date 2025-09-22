import os
import streamlit as st
from groq import Groq

st.set_page_config(page_title="Groq Chatbot", page_icon="🤖", layout="centered")

def init_state():
    if "conversation" not in st.session_state:
        st.session_state.conversation = [
            {"role": "system", "content": "You are a helpful assistant."}
        ]

def get_api_key():
    # prefer Streamlit secrets, then environment variable (useful for local testing)
    if "Groq_api_key" in st.secrets:
        return st.secrets["Groq_api_key"]
    return os.environ.get("GROQ_API_KEY")

def make_client(api_key):
    try:
        return Groq(api_key=api_key)
    except Exception as e:
        st.error(f"Failed to initialize Groq client: {e}")
        return None

def query_groq(client, messages):
    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=messages,
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error calling Groq API: {e}"

def main():
    st.title("🤖 Groq Chatbot")
    st.caption("Multi-turn chat using the Groq SDK and Streamlit")

    init_state()
    api_key = get_api_key()

    # controls row
    c1, c2 = st.columns([1, 4])
    with c1:
        if st.button("Clear conversation"):
            st.session_state.conversation = [{"role":"system","content":"You are a helpful assistant."}]
    with c2:
        user_input = st.chat_input("Type your message...")

    # lazy client init
    client = None
    if api_key:
        client = make_client(api_key)
    else:
        st.info("No Groq API key found. Add `Groq_api_key` in Streamlit Secrets (recommended) or set `GROQ_API_KEY` env var for local testing.")

    if user_input:
        # append user message
        st.session_state.conversation.append({"role":"user","content":user_input})

        if client is None:
            st.session_state.conversation.append({"role":"assistant","content":"(Groq API key not configured.)"})
        else:
            with st.spinner("Thinking..."):
                reply = query_groq(client, st.session_state.conversation)
            st.session_state.conversation.append({"role":"assistant","content":reply})

    # Render chat
    for msg in st.session_state.conversation:
        if msg["role"] == "user":
            with st.chat_message("user"):
                st.write(msg["content"])
        elif msg["role"] == "assistant":
            with st.chat_message("assistant"):
                st.write(msg["content"])

    # optional: show raw history for debugging
    if st.checkbox("Show raw conversation history (debug)"):
        st.json(st.session_state.conversation)

if __name__ == "__main__":
    main()

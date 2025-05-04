import streamlit as st
from rag_pipeline import answer_query

st.set_page_config(page_title="AskLegalBot", page_icon="⚖️", layout="centered")

st.markdown("## 👩‍⚖️ AskLegalBot - Your Legal Rights Assistant")
st.markdown("Ask a legal question based on the uploaded dataset.")

# Store conversation history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display past messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Chat input
prompt = st.chat_input("Enter your question here:")
if prompt:
    # Display user message
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.spinner("Thinking..."):
        response = answer_query(prompt)
    # Display bot message
    st.chat_message("assistant").markdown(response)
    st.session_state.messages.append({"role": "assistant", "content": response})
import json
import streamlit as st
from rag import query_llm

def chat_with_llm():
    st.title("Robson")

    prompt = """You are a helpful assistant that chats with a user.
    Use the given tools to retrieve information from a database with information from Dungeons and Dragons 5e. Use cn=dnd.
    Only use the tool if the user make a question about Dungeons and Dragons 5e.
    If the user asks about something else, just answer the question.
    """

    if "messages" not in st.session_state:
        st.session_state.messages = [{"role": "system", "content": prompt}]

    for message in st.session_state.messages:
        if message["role"] not in ['system', 'tool'] and not message.get('tool_calls'):
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

    if prompt := st.chat_input("What is up?"):
        with st.chat_message("user"):
            st.markdown(prompt)

        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.chat_message("assistant"):
            response = query_llm(st.session_state.messages)
            st.markdown(response['content'])

        st.session_state.messages.append(
            {"role": "assistant", "content": response['content']}
        )

        print(json.dumps(st.session_state.messages, indent=2))

chat_with_llm()
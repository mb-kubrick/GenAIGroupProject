import streamlit as st
from agent_presentation import call_agent

if 'messages' not in st.session_state:
    st.session_state.messages = []

st.title('Investment Management Ai Assistant')
st.markdown('Ask any question about the stock market')

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


if query := st.chat_input("Your Message"):

    with st.chat_message("user"):
        st.markdown(query)

    resp, agent_executor = call_agent(query)
    
    st.session_state.messages.append({"role": "user", "content": query})
    st.session_state.messages.append({"role": "assistant", "content": resp})

    with st.chat_message("assistant"):
        st.markdown(resp)


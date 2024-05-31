import streamlit as st

if 'messages' not in st.session_state:
    st.session_state.messages = []

st.title('Investment Management Chatbot')
st.markdown('Answer any question about the stock market')

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Your message"):

    with st.chat_message("user"):
        st.markdown(prompt)
    
    #answer

    st.session_state.messages.append({"role": "user", "content": prompt})
    #st.session_state.messages.append({"role": "assistant", "content": answer})


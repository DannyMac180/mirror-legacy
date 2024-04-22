import streamlit as st
from lang_programs import LangChainProgram

st.title('Mirror')

# Initialize LangChainProgram instance
lang_chain_program = LangChainProgram()

# Initialize chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Display chat messages from history on app rerun
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("What is up?"):
    # Add user message to chat history of UI
    st.session_state.chat_history.append({"role": "user", "content": prompt})
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)
        
    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        with st.spinner("Reasoning..."):
            try:
                # Create an empty container for the streaming response
                response_container = st.empty()
                
                # Stream the response chunks
                for chunk in lang_chain_program.invoke_chain(prompt, st.session_state.chat_history):
                    response_container.markdown(chunk)
                
                # Append the full response to the chat history of the UI
                st.session_state.chat_history.append({"role": "assistant", "content": chunk})
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
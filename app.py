import streamlit as st
from lang_programs import LangChainProgram

st.title('Mirror')

# Display a sidebar to select the LLM provider
llm_provider = st.sidebar.selectbox("Select LLM Provider", ["lm-studio", "groq"])

# Initialize LangChainProgram instance and store it in the session state
if "lang_chain_program" not in st.session_state:
    st.session_state.lang_chain_program = LangChainProgram(llm_provider)
else:
    # Update the LangChainProgram instance if the llm_provider has changed
    if st.session_state.lang_chain_program.llm_provider != llm_provider:
        st.session_state.lang_chain_program = LangChainProgram(llm_provider)

# Display chat messages from LangChainProgram's memory
for message in st.session_state.lang_chain_program.memory.messages:
    with st.chat_message(message.type):
        st.markdown(message.content)

if prompt := st.chat_input("What is up?"):
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)
        
    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        with st.spinner("Reasoning..."):
            try:
                # Create an empty container for the streaming response
                response_container = st.empty()
                                
                response = st.session_state.lang_chain_program.invoke_chat(prompt)
                
                # Stream the response chunks
                for chunk in response:
                    response_container.markdown(chunk)
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
import streamlit as st

def show_ai_chatbot():
    st.markdown('<h2 style="font-size: 2.5rem; margin-bottom: 1rem; padding-bottom: 0.5rem;">🤖 Health Assistant</h2>', unsafe_allow_html=True)
    st.markdown("""
        <div class="info-box">
            <p>I'm not a licensed medical professional, but I can provide general information and answer questions about various health topics. Keep in mind that I'm not capable of diagnosing medical conditions or providing personalized advice.<br>If you have a specific concern or question, I'll do my best to:<br>1. Provide general information on the topic<br>2. Offer suggestions for further research or consultation with a healthcare professional</p>
        </div>
    """, unsafe_allow_html=True)
    
    try:
        from chatbot import generate_response
        
        # Initialize messages in session state
        if "messages" not in st.session_state:
            st.session_state["messages"] = []

        # Input box and send button
        prompt = st.chat_input("Ask your query here...")
        if prompt:
            # Add user message immediately
            st.session_state["messages"].append({"role": "user", "content": prompt})
            st.rerun()

        # Chat window (show all messages)
        for msg in st.session_state["messages"]:
            if msg["role"] == "user":
                st.chat_message("user").write(msg["content"])
            else:
                st.chat_message("assistant").write(msg["content"])

        # If the last message is from the user, get bot response
        if st.session_state["messages"] and st.session_state["messages"][-1]["role"] == "user":
            with st.spinner("Thinking..."):
                try:
                    # Get the last user message
                    last_user_message = st.session_state["messages"][-1]["content"]
                    
                    # Generate response using Groq
                    response = generate_response(last_user_message)
                    
                    # Add assistant response to messages
                    st.session_state["messages"].append({"role": "assistant", "content": response})
                    st.rerun()
                except Exception as e:
                    error_msg = f"Error communicating with Groq: {str(e)}"
                    st.error(error_msg)
                    st.session_state["messages"].append({"role": "assistant", "content": error_msg})
                    st.rerun()
    except Exception as e:
        st.error(f"Error initializing AI Chatbot: {str(e)}")
        st.info("This feature requires Groq API key to be configured in Streamlit secrets.")

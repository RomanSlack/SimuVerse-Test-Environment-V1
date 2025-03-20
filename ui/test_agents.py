import streamlit as st
import os
from time import sleep
from dotenv import load_dotenv
from modules.framework import create_agent
from src.agent_manager import create_agent_with_llm

def generate_markdown(conversation_log):
    """Helper function to generate markdown for the conversation log."""
    md = ""
    for speaker, msg in conversation_log:
        md += f"**{speaker}:** {msg}\n\n"
    return md

# Load API key from .env
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
hf_api_key = os.getenv("HUGGINGFACE_API_KEY")
claude_api_key = os.getenv("CLAUDE_API_KEY")
if not openai_api_key or not claude_api_key:
    st.error("API Key not found in environment variables.")
    st.stop()

st.title("Multi-Agent Conversation Test")
st.write("This app demonstrates the integration between framework.py and agent_manager.py")

# Test type selection
test_type = st.radio(
    "Select test type:",
    ["Framework Agents Only", "Integrated Agent Manager", "Compare Both"]
)

if test_type == "Framework Agents Only" or test_type == "Compare Both":
    st.subheader("Framework Agents")
    
    # Create two agents with framework.py
    james_framework = create_agent(
        provider="openai",
        name="James",
        api_key=openai_api_key,
        model="gpt-4o-mini",
        system_prompt=(
            "You are James, a friendly conversation partner who is a 20 yr old male. "
            "Please respond naturally in a conversational tone and limit your reply to no more than 2 sentences."
        )
    )
    
    jade_framework = create_agent(
        provider="claude",
        name="Jade",
        api_key=claude_api_key,
        model="claude-3-5-haiku-20241022",
        system_prompt=(
            "You are Jade, an engaging conversation expert who is a 20 yr old female."
            "Respond in a concise, human-like manner using no more than 2 sentences."
        )
    )

if test_type == "Integrated Agent Manager" or test_type == "Compare Both":
    st.subheader("Integrated Agent Manager")
    
    # Create two agents with the integrated agent_manager.py
    james_integrated = create_agent_with_llm(
        agent_id=1,
        name="James",
        provider="openai",
        api_key=openai_api_key,
        model="gpt-4o-mini",
        system_prompt=(
            "You are James, a friendly conversation partner who is a 20 yr old male. "
            "Please respond naturally in a conversational tone and limit your reply to no more than 2 sentences."
        )
    )
    
    jade_integrated = create_agent_with_llm(
        agent_id=2,
        name="Jade",
        provider="claude",
        api_key=claude_api_key,
        model="claude-3-5-haiku-20241022",
        system_prompt=(
            "You are Jade, an engaging conversation expert who is a 20 yr old female."
            "Respond in a concise, human-like manner using no more than 2 sentences."
        )
    )

# Input for the initial message
initial_message = st.text_input(
    "Initial message:", 
    value="Have a conversation with your partner, begin by talking about your lives"
)

# Number of rounds (default is 3 rounds)
num_rounds = st.number_input("Number of rounds", min_value=1, max_value=10, value=3, step=1)

if st.button("Start Conversation") and initial_message:
    # Framework agents conversation (if selected)
    if test_type == "Framework Agents Only" or test_type == "Compare Both":
        with st.expander("Framework Agents Conversation", expanded=True):
            conversation_log = []
            chat_placeholder = st.empty()
            
            # Start conversation: first response from james
            response = james_framework(initial_message)
            conversation_log.append(("James", response))
            chat_placeholder.markdown(generate_markdown(conversation_log))
            sleep(1)  # Optional: small pause for realism
            
            # jade responds to james's reply
            response = jade_framework(response)
            conversation_log.append(("Jade", response))
            chat_placeholder.markdown(generate_markdown(conversation_log))
            sleep(1)
            
            # Continue for remaining rounds
            for i in range(num_rounds - 1):
                response = james_framework(response)
                conversation_log.append(("James", response))
                chat_placeholder.markdown(generate_markdown(conversation_log))
                sleep(1)
                
                response = jade_framework(response)
                conversation_log.append(("Jade", response))
                chat_placeholder.markdown(generate_markdown(conversation_log))
                sleep(1)
                
            # Display whether agents want to move
            st.write("**Movement Check:**")
            st.write(f"James wants to move: {james_framework.wants_to_move()}")
            st.write(f"Jade wants to move: {jade_framework.wants_to_move()}")
    
    # Integrated agents conversation (if selected)
    if test_type == "Integrated Agent Manager" or test_type == "Compare Both":
        with st.expander("Integrated Agent Manager Conversation", expanded=True):
            conversation_log = []
            chat_placeholder = st.empty()
            
            # Start conversation: first response from james
            response = james_integrated.generate_response(initial_message)
            conversation_log.append(("James", response))
            chat_placeholder.markdown(generate_markdown(conversation_log))
            sleep(1)  # Optional: small pause for realism
            
            # jade responds to james's reply
            response = jade_integrated.generate_response(response)
            conversation_log.append(("Jade", response))
            chat_placeholder.markdown(generate_markdown(conversation_log))
            sleep(1)
            
            # Continue for remaining rounds
            for i in range(num_rounds - 1):
                response = james_integrated.generate_response(response)
                conversation_log.append(("James", response))
                chat_placeholder.markdown(generate_markdown(conversation_log))
                sleep(1)
                
                response = jade_integrated.generate_response(response)
                conversation_log.append(("Jade", response))
                chat_placeholder.markdown(generate_markdown(conversation_log))
                sleep(1)
                
            # Display whether agents want to move
            st.write("**Movement Check:**")
            st.write(f"James wants to move: {james_integrated.wants_to_move()}")
            st.write(f"Jade wants to move: {jade_integrated.wants_to_move()}")
            
            # Display state information
            st.write("**Agent States:**")
            st.write(f"James state: {james_integrated.get_state()['state']}")
            st.write(f"Jade state: {jade_integrated.get_state()['state']}")
            
            # Display memory and personality settings
            st.write("**Memory & Personality:**")
            st.write(f"James memory enabled: {james_integrated.get_state()['Memory Enabled']}")
            st.write(f"James personality strength: {james_integrated.get_state()['Personality Strength']}")
            st.write(f"Jade memory enabled: {jade_integrated.get_state()['Memory Enabled']}")
            st.write(f"Jade personality strength: {jade_integrated.get_state()['Personality Strength']}")
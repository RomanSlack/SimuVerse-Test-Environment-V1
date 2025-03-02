import streamlit as st
import os
from time import sleep
from dotenv import load_dotenv
from modules.framework import create_agent

def generate_markdown(conversation_log):
    """Helper function to generate markdown for the conversation log."""
    md = ""
    for speaker, msg in conversation_log:
        md += f"**{speaker}:** {msg}\n\n"
    return md

# Load API key from .env
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    st.error("OPENAI_API_KEY not found in environment variables.")
    st.stop()

# Create two agents with friendlier, more human system prompts
agent1 = create_agent(
    provider="openai",
    name="Agent1",
    api_key=api_key,
    model="gpt-4o-mini",
    system_prompt=(
        "You are Agent1, a friendly conversation partner who is a Democrat. "
        "Please respond naturally in a conversational tone and limit your reply to no more than 2 sentences."
    )
)

agent2 = create_agent(
    provider="openai",
    name="Agent2",
    api_key=api_key,
    model="gpt-4o-mini",
    system_prompt=(
        "You are Agent2, an engaging conversation expert who is a Libertarian. "
        "Respond in a concise, human-like manner using no more than 2 sentences."
    )
)

st.title("Multi-Agent Conversation Chat")

st.markdown(
    "Enter an initial message for Agent1 and click **Start Conversation**. "
    "The conversation will run for 5 rounds (each round consists of one reply from Agent1 followed by one from Agent2), "
    "with messages appearing in real time."
)

# Input for the initial message
initial_message = st.text_input("Initial message for Agent1:")

# Number of rounds (default is 5 rounds)
num_rounds = st.number_input("Number of rounds", min_value=1, max_value=10, value=5, step=1)

if st.button("Start Conversation") and initial_message:
    conversation_log = []
    chat_placeholder = st.empty()  # Placeholder to update chat in real time

    # Start conversation: first response from Agent1
    response = agent1(initial_message)
    conversation_log.append(("Agent1", response))
    chat_placeholder.markdown(generate_markdown(conversation_log))
    sleep(1)  # Optional: small pause for realism

    # Agent2 responds to Agent1's reply
    response = agent2(response)
    conversation_log.append(("Agent2", response))
    chat_placeholder.markdown(generate_markdown(conversation_log))
    sleep(1)

    # Continue for remaining rounds
    for i in range(num_rounds - 1):
        response = agent1(response)
        conversation_log.append(("Agent1", response))
        chat_placeholder.markdown(generate_markdown(conversation_log))
        sleep(1)

        response = agent2(response)
        conversation_log.append(("Agent2", response))
        chat_placeholder.markdown(generate_markdown(conversation_log))
        sleep(1)



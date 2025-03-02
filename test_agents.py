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
openai_api_key = os.getenv("OPENAI_API_KEY")
hf_api_key=os.getenv("HUGGINGFACE_API_KEY")
claude_api_key=os.getenv("CLAUDE_API_KEY")
if not openai_api_key or not hf_api_key:
    st.error("API Key not found in environment variables.")
    st.stop()

# Create two agents with friendlier, more human system prompts
james = create_agent(
    provider="openai",
    name="James",
    api_key=openai_api_key,
    model="gpt-4o-mini",
    system_prompt=(
        "You are James, a friendly conversation partner who is a 20 yr old male. "
        "Please respond naturally in a conversational tone and limit your reply to no more than 2 sentences."
    )
)

jade = create_agent(
    provider="claude",
    name="Jade",
    api_key=claude_api_key,
    model="claude-3-5-haiku-20241022",
    system_prompt=(
        "You are Jade, an engaging conversation expert who is a 20 yr old female."
        "Respond in a concise, human-like manner using no more than 2 sentences."
    )
)

st.title("Multi-Agent Conversation Chat")

st.markdown(
    "Enter an initial message for james and click **Start Conversation**. "
    "The conversation will run for 5 rounds (each round consists of one reply from james followed by one from jade), "
    "with messages appearing in real time."
)

# Input for the initial message (Have a conversation with Jade, begin by talking about your lives)
initial_message = st.text_input("Initial message for james:", value="Have a conversation with Jade, begin by talking about your lives")

# Number of rounds (default is 5 rounds)
num_rounds = st.number_input("Number of rounds", min_value=1, max_value=10, value=1, step=1)

if st.button("Start Conversation") and initial_message:
    conversation_log = []
    chat_placeholder = st.empty()  # Placeholder to update chat in real time

    # Start conversation: first response from james
    response = james(initial_message)
    conversation_log.append(("James", response))
    chat_placeholder.markdown(generate_markdown(conversation_log))
    sleep(1)  # Optional: small pause for realism

    # jade responds to james's reply
    response = jade(response)
    conversation_log.append(("Jade", response))
    chat_placeholder.markdown(generate_markdown(conversation_log))
    sleep(1)

    # Continue for remaining rounds
    for i in range(num_rounds - 1):
        response = james(response)
        conversation_log.append(("james", response))
        chat_placeholder.markdown(generate_markdown(conversation_log))
        sleep(1)

        response = jade(response)
        conversation_log.append(("jade", response))
        chat_placeholder.markdown(generate_markdown(conversation_log))
        sleep(1)



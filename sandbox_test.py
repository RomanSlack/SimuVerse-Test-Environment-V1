import os
from dotenv import load_dotenv
from modules.framework import create_framework

# Load environment variables from .env file
load_dotenv()

# Retrieve the API key from the environment variable
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OPENAI_API_KEY not found in environment variables.")

# Create the framework with desired parameters
framework = create_framework(
    provider="openai",              # or "local"
    api_key=api_key,                # Retrieved from .env
    model="gpt-3.5-turbo",
    system_prompt="You are a helpful assistant.",
    agent_names=["Agent1", "Agent2"]
)

# Run the agents with some test input
user_input = "Tell me a joke."
responses = framework.run_agents(user_input)
for agent, reply in responses.items():
    print(f"{agent}: {reply}")

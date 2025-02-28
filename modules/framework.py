#!/usr/bin/env python3
"""
Multi-Agent Framework Module

This module provides a multi-agent framework that supports various LLM providers.
You can import the `create_framework` function to instantiate and control the framework
in your test files. Currently, it supports the 'openai' and 'local' providers, and you
can easily add more by implementing the BaseLLM interface.
"""

import abc
from typing import Optional, List, Dict


# =============================================================================
# Base Interface for LLM Providers
# =============================================================================

class BaseLLM(abc.ABC):
    """
    Abstract base class for LLM backends.
    Each subclass must implement the `generate` method.
    """

    @abc.abstractmethod
    def generate(self, prompt: str) -> str:
        """
        Generate a response based on the given prompt.
        """
        pass


# =============================================================================
# OpenAI ChatGPT Implementation
# =============================================================================

class OpenAIChatGPT(BaseLLM):
    """
    LLM interface for the OpenAI ChatGPT API using the updated client library.
    """

    def __init__(self, api_key: str, model: str = "gpt-3.5-turbo"):
        from openai import OpenAI
        self.api_key = api_key
        self.model = model
        # Initialize the OpenAI client with the provided API key
        self.client = OpenAI(api_key=api_key)

    def generate(self, prompt: str) -> str:
        """
        Generate a response using the OpenAI ChatCompletion API with the updated usage.
        Instead of indexing the response as a dictionary, we now use attribute access.
        """
        response = self.client.chat.completions.create(
            messages=[{"role": "system", "content": prompt}],
            model=self.model,
        )
        # Use attribute access instead of dictionary subscripting
        return response.choices[0].message.content


# =============================================================================
# Local Open Source Model Implementation
# =============================================================================

class LocalLLM(BaseLLM):
    """
    LLM interface for a local open source model using Hugging Face transformers.
    """

    def __init__(self, model_name: str = "gpt2"):
        from transformers import pipeline
        # Create a text-generation pipeline for the model
        self.generator = pipeline("text-generation", model=model_name)

    def generate(self, prompt: str) -> str:
        """
        Generate a response using the local model.
        Adjust parameters like max_length as needed.
        """
        results = self.generator(prompt, max_length=100, num_return_sequences=1)
        return results[0]["generated_text"]


# =============================================================================
# Agent Class
# =============================================================================

class Agent:
    """
    Represents an individual agent with a name, LLM backend, and a system prompt.
    """

    def __init__(self, name: str, llm: BaseLLM, system_prompt: str):
        self.name = name
        self.llm = llm
        self.system_prompt = system_prompt

    def run(self, user_input: str) -> str:
        """
        Combine the system prompt with user input and generate a response.
        """
        # Construct the final prompt for the LLM
        prompt = f"{self.system_prompt}\nUser: {user_input}\n{self.name}:"
        response = self.llm.generate(prompt)
        return response


# =============================================================================
# Multi-Agent Framework
# =============================================================================

class MultiAgentFramework:
    """
    Framework to manage and run multiple agents.
    """

    def __init__(self):
        self.agents: Dict[str, Agent] = {}

    def add_agent(self, agent: Agent):
        """
        Add an agent to the framework.
        """
        self.agents[agent.name] = agent

    def run_agents(self, user_input: str) -> Dict[str, str]:
        """
        Run all agents with the provided user input and collect their responses.

        Returns:
            A dictionary mapping agent names to their responses.
        """
        responses = {}
        for name, agent in self.agents.items():
            responses[name] = agent.run(user_input)
        return responses


# =============================================================================
# Function to Create the Framework
# =============================================================================

def create_framework(provider: str,
                     api_key: Optional[str] = None,
                     model: Optional[str] = None,
                     system_prompt: str = "You are a helpful assistant.",
                     agent_names: Optional[List[str]] = None) -> MultiAgentFramework:
    """
    Create and return a MultiAgentFramework instance with agents configured based on the parameters.

    Parameters:
        provider (str): The LLM provider to use ('openai' or 'local').
        api_key (Optional[str]): API key for the provider if required (e.g., for OpenAI).
        model (Optional[str]): The model name to use. Defaults depend on the provider.
        system_prompt (str): The system prompt for the agents.
        agent_names (Optional[List[str]]): A list of agent names to create. Defaults to ["Agent1", "Agent2"].

    Returns:
        MultiAgentFramework: The configured multi-agent framework.
    """
    # Set default agent names if not provided
    if agent_names is None:
        agent_names = ["Agent1", "Agent2"]

    # Instantiate the appropriate LLM backend
    if provider == "openai":
        if not api_key:
            raise ValueError("API key is required for the OpenAI provider")
        llm_instance = OpenAIChatGPT(api_key=api_key,
                                     model=model if model else "gpt-3.5-turbo")
    elif provider == "local":
        llm_instance = LocalLLM(model_name=model if model else "gpt2")
    else:
        raise ValueError("Unsupported provider selected. Use 'openai' or 'local'.")

    # Create the multi-agent framework and add agents
    framework = MultiAgentFramework()
    for agent_name in agent_names:
        agent = Agent(name=agent_name, llm=llm_instance, system_prompt=system_prompt)
        framework.add_agent(agent)

    return framework


# =============================================================================
# Optional Test Function (Run when module is executed directly)
# =============================================================================

if __name__ == "__main__":
    # Example usage when running the module directly
    # Adjust these parameters as needed for testing
    test_provider = "local"  # Change to "openai" if using the OpenAI provider
    test_model = "gpt2"      # Use "gpt-3.5-turbo" for OpenAI
    test_system_prompt = "You are a friendly assistant."
    test_agent_names = ["TestAgent1", "TestAgent2"]

    # Create the framework using the test parameters
    framework = create_framework(provider=test_provider,
                                 model=test_model,
                                 system_prompt=test_system_prompt,
                                 agent_names=test_agent_names)

    # Get user input and run the agents
    user_input = input("Enter your message: ")
    responses = framework.run_agents(user_input)
    for agent_name, response in responses.items():
        print(f"\n{agent_name} response:\n{response}\n")

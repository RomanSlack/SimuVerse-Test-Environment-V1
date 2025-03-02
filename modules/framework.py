#!/usr/bin/env python3
"""
Multi-Agent Framework Module

This module provides a multi-agent framework that supports various LLM providers.
You can import the `create_framework` function to instantiate a container of agents or
use `create_agent` to create a single agent instance with its own conversation session.
Currently, it supports the 'openai' and 'local' providers, and you can easily add more
by implementing the BaseLLM interface.
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
        self.client = OpenAI(api_key=api_key)

    def generate(self, prompt: str) -> str:
        """
        Generate a response using the OpenAI ChatCompletion API.
        Instead of subscripting the response, we use attribute access.
        """
        response = self.client.chat.completions.create(
            messages=[{"role": "system", "content": prompt}],
            model=self.model,
        )
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
        self.generator = pipeline("text-generation", model=model_name)

    def generate(self, prompt: str) -> str:
        """
        Generate a response using the local model.
        Adjust parameters like max_length as needed.
        """
        results = self.generator(prompt, max_length=100, num_return_sequences=1)
        return results[0]["generated_text"]


# =============================================================================
# Agent Class with Conversation Session
# =============================================================================

class Agent:
    """
    Represents an individual agent with a name, LLM backend, a system prompt,
    and its own conversation session.
    """

    def __init__(self, name: str, llm: BaseLLM, system_prompt: str):
        self.name = name
        self.llm = llm
        self.system_prompt = system_prompt
        # Conversation history starts with the system prompt
        self.conversation_history = [{"role": "system", "content": system_prompt}]

    def send(self, user_input: str) -> str:
        """
        Append the user message to the conversation, generate a response,
        update the conversation history, and return the agent's reply.
        """
        self.conversation_history.append({"role": "user", "content": user_input})
        prompt = self.build_prompt()
        response = self.llm.generate(prompt)
        self.conversation_history.append({"role": "assistant", "content": response})
        return response

    def build_prompt(self) -> str:
        """
        Build a prompt string by concatenating the conversation history.
        Each message is formatted with a label.
        """
        prompt_lines = []
        for msg in self.conversation_history:
            if msg["role"] == "system":
                prompt_lines.append(f"System: {msg['content']}")
            elif msg["role"] == "user":
                prompt_lines.append(f"User: {msg['content']}")
            elif msg["role"] == "assistant":
                prompt_lines.append(f"{self.name}: {msg['content']}")
        # Append the agent name to signal that a response is expected
        prompt_lines.append(f"{self.name}:")
        return "\n".join(prompt_lines)

    def clear_history(self):
        """
        Clear conversation history, keeping only the system prompt.
        """
        self.conversation_history = [{"role": "system", "content": self.system_prompt}]

    def __call__(self, user_input: str) -> str:
        """
        Allow the agent instance to be called like a function.
        """
        return self.send(user_input)


# =============================================================================
# Multi-Agent Framework Container (Optional)
# =============================================================================

class MultiAgentFramework:
    """
    Framework container to manage multiple agents.
    """

    def __init__(self):
        self.agents: Dict[str, Agent] = {}

    def add_agent(self, agent: Agent):
        """
        Add an agent to the container.
        """
        self.agents[agent.name] = agent

    def get_agent(self, name: str) -> Agent:
        """
        Retrieve an agent by name.
        """
        return self.agents.get(name)

    def run_agents(self, user_input: str) -> Dict[str, str]:
        """
        Run all agents with the provided input and collect their responses.
        """
        responses = {}
        for name, agent in self.agents.items():
            responses[name] = agent.send(user_input)
        return responses


# =============================================================================
# Factory Functions to Create Agents or a Framework
# =============================================================================

def create_agent(provider: str,
                 name: str,
                 api_key: Optional[str] = None,
                 model: Optional[str] = None,
                 system_prompt: str = "You are a helpful assistant.") -> Agent:
    """
    Create and return a single Agent instance.

    Example usage:
        agent1 = create_agent(provider="openai", name="Agent1", api_key=API_KEY, model="gpt-3.5-turbo")
    """
    if provider == "openai":
        if not api_key:
            raise ValueError("API key is required for the OpenAI provider")
        llm_instance = OpenAIChatGPT(api_key=api_key, model=model if model else "gpt-3.5-turbo")
    elif provider == "local":
        llm_instance = LocalLLM(model_name=model if model else "gpt2")
    else:
        raise ValueError("Unsupported provider selected. Use 'openai' or 'local'.")
    return Agent(name=name, llm=llm_instance, system_prompt=system_prompt)


def create_framework(provider: str,
                     api_key: Optional[str] = None,
                     model: Optional[str] = None,
                     system_prompt: str = "You are a helpful assistant.",
                     agent_names: Optional[List[str]] = None) -> MultiAgentFramework:
    """
    Create and return a MultiAgentFramework instance with multiple agents.
    """
    if agent_names is None:
        agent_names = ["Agent1", "Agent2"]
    framework = MultiAgentFramework()
    for agent_name in agent_names:
        agent = create_agent(provider=provider,
                             name=agent_name,
                             api_key=api_key,
                             model=model,
                             system_prompt=system_prompt)
        framework.add_agent(agent)
    return framework


# =============================================================================
# Optional Test Function (Run when module is executed directly)
# =============================================================================

if __name__ == "__main__":
    # Example usage with a multi-agent container (using local provider for testing)
    test_provider = "local"  # Change to "openai" if desired
    test_model = "gpt2"  # Use "gpt-3.5-turbo" for OpenAI
    test_system_prompt = "You are a friendly assistant."
    test_agent_names = ["TestAgent1", "TestAgent2"]

    framework = create_framework(provider=test_provider,
                                 model=test_model,
                                 system_prompt=test_system_prompt,
                                 agent_names=test_agent_names)

    user_input = input("Enter your message: ")
    responses = framework.run_agents(user_input)
    for agent_name, response in responses.items():
        print(f"\n{agent_name} response:\n{response}\n")

#!/usr/bin/env python3
"""
Multi-Agent Framework Module

This module provides a multi-agent framework that supports various LLM providers.
You can import the `create_framework` function to instantiate a container of agents or
use `create_agent` to create a single agent instance with its own conversation session.
Currently, it supports the 'openai', 'local' (Hugging Face Transformers pipeline),
'ollama', 'huggingface' (Hugging Face Inference API), and 'claude' (Anthropic Claude API) providers.
You can easily add more by implementing the BaseLLM interface.
"""

import abc
import re
import logging
from typing import Optional, List, Dict, Tuple


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
        Parse the prompt to identify system, user, and assistant roles.
        """
        # Process the prompt to extract conversation parts
        messages = []
        
        # Extract the system instruction part
        if "# Instructions" in prompt:
            parts = prompt.split("# Instructions")
            system_part = parts[1].split("#")[0].strip()
            messages.append({"role": "system", "content": system_part})
        
        # Extract conversation history
        if "# Conversation History" in prompt:
            history_start = prompt.find("# Conversation History")
            history_text = prompt[history_start:].split("\n", 1)[1]
            
            # Parse remaining history into messages
            for line in history_text.split("\n"):
                line = line.strip()
                if not line:  # Skip empty lines
                    continue
                    
                if line.startswith("USER:"):
                    messages.append({"role": "user", "content": line[5:].strip()})
                elif line.startswith("SYSTEM:"):
                    # System messages in the history go as user messages with special formatting
                    messages.append({"role": "user", "content": line})
                elif ":" in line and not line.startswith(("1.", "2.", "3.", "4.")):
                    # Assistant messages (with agent name)
                    name_part = line.split(":", 1)[0].strip()
                    content_part = line.split(":", 1)[1].strip()
                    
                    if "respond with" not in content_part:  # Skip the final instruction line
                        messages.append({"role": "assistant", "content": content_part})
        
        # If we have no messages, just send the whole prompt as system message
        if not messages:
            messages = [{"role": "system", "content": prompt}]
            
        # Add a user message at the end to prompt for response if not already there
        if not messages or messages[-1]["role"] != "user":
            messages.append({"role": "user", "content": "Please respond to the conversation."})
            
        # Send to the API
        response = self.client.chat.completions.create(
            messages=messages,
            model=self.model,
            max_tokens=150,  # Limit response length
            temperature=0.7  # Add some variation
        )
        
        # Log for debugging
        logging.debug(f"OpenAI messages: {messages}")
        return response.choices[0].message.content
    
    def __str__(self):
        return f"{self.model}"


# =============================================================================
# Local Open Source Model Implementation (Hugging Face Transformers Pipeline)
# =============================================================================

class LocalLLM(BaseLLM):
    """
    LLM interface for a local open source model using Hugging Face transformers.
    """
    def __init__(self, model_name: str = "gpt2"):
        from transformers import pipeline
        self.model_name = model_name
        self.generator = pipeline("text-generation", model=model_name)

    def generate(self, prompt: str) -> str:
        """
        Generate a response using the local model.
        Adjust parameters like max_length as needed.
        """
        results = self.generator(prompt, max_length=100, num_return_sequences=1)
        return results[0]["generated_text"]
    
    def __str__(self):
        return f"{self.model_name}"

# =============================================================================
# Ollama LLM Implementation
# =============================================================================

class OllamaLLM(BaseLLM):
    """
    LLM interface for a local Ollama deployment.
    This assumes Ollama is running locally and exposes an HTTP API.
    Adjust the endpoint and payload as necessary based on your setup.
    """
    def __init__(self, model: str = "ollama-model"):
        import requests
        self.model = model
        self.endpoint = "http://localhost:11434/api/generate"  # Adjust this URL as needed.
        self.requests = requests

    def generate(self, prompt: str) -> str:
        payload = {
            "model": self.model,
            "prompt": prompt,
            "max_tokens": 200  # Adjust token count as desired.
        }


        response = self.requests.post(self.endpoint, json=payload)
        response.raise_for_status()
        data = response.json()
        # Assume the API returns JSON with a "response" field containing the answer.
        return data.get("response", "")
    
    def __str__(self):
        return f"{self.model}"

# =============================================================================
# Hugging Face Inference API Implementation
# =============================================================================

class HuggingFaceLLM(BaseLLM):
    """
    LLM interface for the Hugging Face Inference API.
    This class uses your Hugging Face API token to call the inference API.
    """
    def __init__(self, api_token: str, model: str = "gpt2"):
        import requests
        self.api_token = api_token
        self.model = model
        self.api_url = f"https://api-inference.huggingface.co/models/{model}"
        self.requests = requests

    def generate(self, prompt: str) -> str:
        headers = {"Authorization": f"Bearer {self.api_token}"}
        payload = {"inputs": prompt}
        response = self.requests.post(self.api_url, headers=headers, json=payload)
        response.raise_for_status()
        data = response.json()
        # For text-generation models, the response is typically a list of dictionaries.
        return data[0]["generated_text"] if isinstance(data, list) and "generated_text" in data[0] else ""

    def __str__(self):
        return f"{self.model}"

# =============================================================================
# Claude API Implementation
# =============================================================================

class ClaudeLLM(BaseLLM):
    """
    LLM interface for Anthropic's Claude API.
    This class uses your Claude API token to call the Anthropic API.
    """
    def __init__(self, api_key: str, model: str = "claude-v1"):
        import anthropic
        self.api_key = api_key
        self.model = model
        self.client = anthropic.Anthropic(api_key=api_key)

    def generate(self, prompt: str) -> str:
        """
        Generate a response using Anthropic's Claude API.
        Claude expects messages in a list format; parse the prompt to build appropriate messages.
        """
        messages = []
        system_message = ""
        
        # Extract the system instruction part
        if "# Instructions" in prompt:
            parts = prompt.split("# Instructions")
            system_part = parts[1].split("#")[0].strip()
            system_message = system_part
            
        # Process the rest of the conversation
        conversation = prompt
        
        # Create a single user message with Claude system prompt embedded
        if system_message:
            prompt_with_system = f"<instructions>{system_message}</instructions>\n\n{conversation}"
        else:
            prompt_with_system = conversation
            
        # Create the message structure
        messages = [{"role": "user", "content": prompt_with_system}]
            
        # Send to the API
        response = self.client.messages.create(
            model=self.model,
            max_tokens=150,  # Limit response length 
            messages=messages
        )
        
        # Extract the content depending on the API response format
        if hasattr(response, 'content') and hasattr(response.content[0], 'text'):
            # Current API format
            return response.content[0].text
        elif hasattr(response, 'content'):
            # Handle content object
            output = str(response.content)
            if 'text="' in output:
                part_after_text = output.split('text="')[1]  # everything after text="
                extracted_text = part_after_text.split('"')[0]  # grab everything until the next "
                return extracted_text
            else:
                return output
                
        # Fallback
        return str(response)
    
    def __str__(self):
        return f"{self.model}"

# =============================================================================
# Agent Class with Conversation Session
# =============================================================================

class Agent:
    """
    Represents an individual agent with a name, LLM backend, a system prompt,
    and its own conversation session.
    """
    def __init__(self, name: str, llm: BaseLLM, system_prompt: str, 
                 memory_enabled: bool = True, personality_strength: float = 0.5):
        self.name = name
        self.llm = llm
        self.system_prompt = system_prompt
        self.conversation_history = [{"role": "system", "content": system_prompt}]
        self.memory_enabled = memory_enabled  # New
        self.personality_strength = personality_strength  # New
        self.action_requests = []  # Store action requests like "move"
        
    # Add setters for new properties
    def set_memory_enabled(self, enabled: bool):
        self.memory_enabled = enabled
        
    def set_personality_strength(self, strength: float):
        self.personality_strength = max(0.0, min(1.0, strength))  # Clamp between 0-1

    # Modify send method to use memory setting and detect movement intent
    def send(self, user_input: str) -> str:
        # Check if this is a system message
        is_system_message = user_input.startswith("[SYSTEM:") 

        if self.memory_enabled:
            # Add to conversation history with proper role
            role = "system" if is_system_message else "user"
            self.conversation_history.append({"role": role, "content": user_input})
            
        # Append special instruction to detect movement requests
        movement_instruction = "\n\nIf you want to move to meet someone new, include the exact text [MOVE] somewhere in your response."
        instructions = "\n\nRemember to stay in character. Keep your response brief (1-2 sentences max)."
        
        # Only add the movement instruction if this isn't a system message
        if not is_system_message:
            prompt = self.build_prompt() + instructions + movement_instruction
        else:
            prompt = self.build_prompt()
        
        # Attempt to get a valid response (up to 3 tries)
        max_attempts = 3
        for attempt in range(max_attempts):
            response = self.llm.generate(prompt)
            
            # Filter and validate the response
            is_valid, filtered_response = self._filter_response(response, is_system_message)
            
            if is_valid:
                response = filtered_response
                break
                
            # If we hit a bad response, add feedback and try again
            if attempt < max_attempts - 1:
                prompt += "\n\nPlease rewrite your response. Keep it very brief (1-2 sentences) and stay in character."
        
        # Check for movement request in the response
        # Only look for [MOVE] tag in non-system responses
        if not is_system_message and "[MOVE]" in response:
            # Add to action requests queue
            self.action_requests.append("move")
            # Clean the response for display (remove the action tag)
            response = response.replace("[MOVE]", "")
        
        # Store only valid responses in the memory
        if self.memory_enabled:
            self.conversation_history.append({"role": "assistant", "content": response})
            
        return response
        
    def _filter_response(self, response: str, is_system_message: bool) -> Tuple[bool, str]:
        """
        Filter and validate agent responses, returning (is_valid, filtered_response)
        """
        # Don't filter system message responses
        if is_system_message:
            return True, response
            
        # Log the original response for debugging
        logging.debug(f"{self.name} original response: {response}")
            
        # Reject if response includes the main system prompt text
        if "You are " + self.name in response:
            logging.info(f"{self.name} response filtered: contains system prompt text")
            return False, response
            
        # Reject if the response has first person description of system prompt abilities
        if any(x in response.lower() for x in ["i am a", "i'm a", "as an ai", "as an assistant", "as your assistant"]):
            logging.info(f"{self.name} response filtered: AI self-reference")
            return False, response
            
        # Reject if the response has wrong name (using parentheses to indicate role playing)
        if "(" in response and ")" in response and not f"({self.name})" in response:
            logging.info(f"{self.name} response filtered: using wrong character name")
            return False, response
            
        # Reject if the response is too long (more than 3 sentences)
        sentences = [s.strip() for s in response.split('.') if s.strip()]
        if len(sentences) > 3:
            # Truncate to 2 sentences
            filtered = '. '.join(sentences[:2]) + '.'
            logging.info(f"{self.name} response truncated from {len(sentences)} to 2 sentences")
            return True, filtered
        
        # Reject if the response is extremely long (more than 200 characters without punctuation)
        if len(response) > 200 and response.count('.') < 2:
            logging.info(f"{self.name} response filtered: too long without proper sentence structure")
            return False, response
            
        # Reject responses that look like the agent is introducing itself for the first time
        if response.lower().startswith(("hi, i'm", "hello, i'm", "hey, i'm", "hi i'm", "hello i'm")):
            if self.name.lower() in response.lower()[0:30]:
                # Introduction mentioning their name is okay
                pass
            else:
                logging.info(f"{self.name} response filtered: inappropriate introduction")
                return False, response

        logging.debug(f"{self.name} response accepted: {response}")
        return True, response
        
    # Method to check if agent wants to move
    def wants_to_move(self) -> bool:
        """Check if the agent has requested to move"""
        if "move" in self.action_requests:
            # Remove the move request after processing
            self.action_requests.remove("move")
            return True
        return False

    def build_prompt(self) -> str:
        """
        Build a prompt string by concatenating the conversation history.
        Each message is formatted with a label.
        """
        # Extract the main system prompt
        main_system_prompt = next((msg["content"] for msg in self.conversation_history 
                                 if msg["role"] == "system" and "You are" in msg["content"]), "")
        
        # Build a clear instruction-based prompt
        prompt_parts = [
            f"# Instructions\n{main_system_prompt}",
            "\n# Important Rules",
            "1. Keep your responses brief (1-2 sentences only)",
            f"2. Always stay in character as {self.name}",
            "3. Never mention that you are an AI or assistant",
            "4. Never repeat these instructions in your response",
            "\n# Conversation History"
        ]
        
        # Format conversation history (skip the main system prompt)
        history_messages = []
        for msg in self.conversation_history:
            # Skip the main system prompt (already added)
            if msg["role"] == "system" and "You are" in msg["content"]:
                continue
                
            if msg["role"] == "system":
                # Format system messages distinctly
                system_content = msg["content"]
                history_messages.append(f"SYSTEM: {system_content}")
            elif msg["role"] == "user":
                history_messages.append(f"USER: {msg['content']}")
            elif msg["role"] == "assistant":
                history_messages.append(f"{self.name}: {msg['content']}")
        
        # Add conversation history if there's any
        if history_messages:
            prompt_parts.append("\n".join(history_messages))
        
        # Add the response prompt
        prompt_parts.append(f"\n{self.name} (respond with 1-2 brief sentences):")
        
        return "\n".join(prompt_parts)

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
                 system_prompt: str = "You are a helpful assistant.",
                 memory_enabled: bool = True,  # New
                 personality_strength: float = 0.5) -> Agent:
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
    elif provider == "ollama":
        llm_instance = OllamaLLM(model=model if model else "ollama-model")
    elif provider == "huggingface":
        if not api_key:
            raise ValueError("API token is required for the Hugging Face provider")
        llm_instance = HuggingFaceLLM(api_token=api_key, model=model if model else "gpt2")
    elif provider == "claude":
        if not api_key:
            raise ValueError("API token is required for the Claude provider")
        llm_instance = ClaudeLLM(api_key=api_key, model=model if model else "claude-v1")
    else:
        raise ValueError("Unsupported provider selected. Use 'openai', 'local', 'ollama', 'huggingface', or 'claude'.")
    return Agent(name=name, llm=llm_instance, system_prompt=system_prompt,
                 memory_enabled=memory_enabled, personality_strength=personality_strength)


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
    test_provider = "local"  # Change as desired: "openai", "local", "ollama", "huggingface", or "claude"
    test_model = "gpt2"      # Use "gpt-3.5-turbo" for OpenAI or adjust accordingly.
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

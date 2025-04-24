from enum import Enum
import random
import logging
import os
import asyncio
import time
from typing import List, Dict, Optional, Union, Tuple, Coroutine, Any, Set
import uuid
from logging.handlers import RotatingFileHandler

from modules.framework import BaseLLM, Agent as FrameworkAgent, create_agent

# Ensure logs directory exists
os.makedirs("logs", exist_ok=True)

# Configuring the logging global var
log_handler = RotatingFileHandler(
    "logs/conversations.log",
    maxBytes=25_000_000,
    backupCount=3
)
log_handler.setFormatter(
    logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
)
LOG = logging.getLogger()
LOG.setLevel(logging.INFO)
LOG.addHandler(log_handler)


class Status(Enum):
    TALKING = "orange"
    THINKING = "purple"
    IDLE = "skyblue"


class Message:
    """
    Messages sent between agents
    """
    def __init__(self, sender_id: int, content: str, recipient_id: int = None, conversation_id: str = None, is_system: bool = False, requires_response: bool = True):
        self.sender_id = sender_id
        self.content = content
        self.recipient_id = recipient_id
        # Conversation ID to track specific conversations between agent pairs
        # If not provided, it will be generated when added to a conversation
        self.conversation_id = conversation_id
        # Flag to indicate if this is a system message
        self.is_system = is_system or sender_id == 0
        # Flag to indicate if this message requires a response
        # System notifications like movement don't need responses
        self.requires_response = requires_response

    def __str__(self):
        # Prints the message in readable format
        conv_id = f" [Conv: {self.conversation_id}]" if self.conversation_id else ""
        system_tag = " [SYSTEM]" if self.is_system else ""
        return f"{self.sender_id} -> {self.recipient_id}{conv_id}{system_tag}: {self.content}"

    def set_recipient(self, id: int):
        # Sets the reciever of a message
        self.recipient_id = id
        
    def set_conversation_id(self, conv_id: str):
        # Sets the conversation ID
        self.conversation_id = conv_id


class Conversation:
    """
    Tracks a conversation between specific agents
    """
    def __init__(self, agent_ids: Set[int], conversation_id: str = None):
        # The conversation ID is a unique identifier for this conversation
        self.conversation_id = conversation_id or str(uuid.uuid4())
        # The set of agent IDs participating in this conversation
        self.agent_ids = agent_ids
        # The messages in this conversation
        self.messages = []
        # When the conversation was created
        self.created_at = time.time()
        # The last time a message was added
        self.last_updated = self.created_at
        # The round count in this conversation
        self.round_count = 0
        # Additional metadata
        self.metadata = {}
        
    def add_message(self, message: Message):
        # Set conversation ID on the message if not already set
        if not message.conversation_id:
            message.set_conversation_id(self.conversation_id)
        # Add the message to the conversation
        self.messages.append(message)
        # Update last updated time using standard time module
        self.last_updated = time.time()
        
    def get_messages(self) -> List[Message]:
        # Return the messages in this conversation
        return self.messages
    
    def increment_round(self):
        # Increment the round count
        self.round_count += 1
        
    def is_between(self, agent_id1: int, agent_id2: int) -> bool:
        # Check if this conversation is between the specified agents
        return {agent_id1, agent_id2}.issubset(self.agent_ids)
    
    def get_conversation_partner(self, agent_id: int) -> Optional[int]:
        # If this is a two-person conversation, get the other participant
        if len(self.agent_ids) == 2 and agent_id in self.agent_ids:
            partner_id = list(self.agent_ids - {agent_id})[0]
            return partner_id
        return None


class BaseAgent:
    """
    Basic agent with identifiers
    """
    def __init__(self, agent_id: int, name: str):
        self.id = agent_id
        self.name = name
        self.convo_history = []
        self.last_msg = None
        self.state = Status.IDLE
        # Integration with framework LLM
        self.framework_agent = None
        # Track thinking state for UI
        self.thinking = False
        # Track conversations the agent is participating in
        self.conversations = {}  # conversation_id -> Conversation
        # Current active conversation ID
        self.active_conversation_id = None
        # Conversation metadata for improved history tracking
        self.conversation_metadata = {}

    def __repr__(self):
        return (
            f"{self.id}: {self.name}\n\t"
            f"{self.convo_history}\n\t"
            f"{self.state}\n\t"
        )

    def get_id_pair(self):
        return f"{self.id}: {self.name}"

    def add_to_conversation(self, msg: Message, conversation_id: str = None) -> str:
        """
        Add a message to a conversation, creating a new one if necessary
        Returns the conversation ID
        """
        # If conversation ID is provided, try to get the existing conversation
        if conversation_id and conversation_id in self.conversations:
            conversation = self.conversations[conversation_id]
        # If message has a conversation ID, try to get that conversation
        elif msg.conversation_id and msg.conversation_id in self.conversations:
            conversation = self.conversations[msg.conversation_id]
            conversation_id = msg.conversation_id
        # Otherwise, check if we have an existing conversation with this agent
        elif msg.sender_id != self.id and any(
            conv.is_between(self.id, msg.sender_id) for conv in self.conversations.values()
        ):
            # Find the existing conversation with this agent
            for conv in self.conversations.values():
                if conv.is_between(self.id, msg.sender_id):
                    conversation = conv
                    conversation_id = conv.conversation_id
                    break
        # If still no conversation, create a new one
        else:
            conversation_id = str(uuid.uuid4())
            conversation = Conversation(
                agent_ids={self.id, msg.sender_id if msg.sender_id != self.id else msg.recipient_id},
                conversation_id=conversation_id
            )
            self.conversations[conversation_id] = conversation
        
        # Add the message to the conversation
        conversation.add_message(msg)
        
        # Set the active conversation ID
        self.active_conversation_id = conversation_id
        
        return conversation_id
    
    def receive_message(self, msg: Message):
        # Store received message in conversation system
        conversation_id = self.add_to_conversation(msg)
        
        # Also keep in legacy history for backward compatibility
        self.convo_history.append(msg)
        self.last_msg = msg
        
        # Logs message
        logging.info(str(msg))

        # Only generate a response if the message requires one
        if msg.requires_response:
            # Pass the conversation_id to maintain context
            return self.generate_response(msg.content, conversation_id)
        else:
            # For system messages that don't need a response, return None
            logging.info(f"No response required for system message: {msg.content}")
            return None
    
    async def receive_message_async(self, msg: Message) -> Optional[str]:
        """Async version of receive_message"""
        # Store received message in conversation system
        conversation_id = self.add_to_conversation(msg)
        
        # Also keep in legacy history for backward compatibility
        self.convo_history.append(msg)
        self.last_msg = msg
        
        # Logs message
        logging.info(str(msg))

        # Only generate a response if the message requires one
        if not msg.requires_response:
            logging.info(f"No response required for system message: {msg.content}")
            return None

        # Set thinking state for UI
        self.thinking = True
        self.set_state(Status.THINKING)
        
        # Generate a response asynchronously - pass the conversation_id
        response = await self.generate_response_async(msg.content, conversation_id)
        
        # Reset thinking state
        self.thinking = False
        self.set_state(Status.TALKING)
        
        return response

    def generate_response(self, message: str, conversation_id: str = None) -> str:
        """
        Base response generation.
        Can take an optional conversation_id to maintain context for multiple conversations.
        """
        # Store the conversation ID we're currently responding to
        self.active_conversation_id = conversation_id
        
        if self.framework_agent:
            # Use the framework agent if available
            # Add conversation context to the message to help the LLM understand which conversation this belongs to
            if conversation_id:
                # Get the conversation partner for context
                conversation = self.get_conversation(conversation_id)
                if conversation:
                    partner_id = conversation.get_conversation_partner(self.id)
                    if partner_id is not None:
                        # Find the partner's name
                        partner_name = f"Agent_{partner_id}"  # Default
                        for name, agent in self.get_conversation_metadata().get("agent_lookup", {}).items():
                            if hasattr(agent, 'id') and agent.id == partner_id:
                                partner_name = name
                                break
                                
                        # Add conversation context (subtly)
                        message_with_context = f"[Conversation with {partner_name}] {message}"
                        return self.framework_agent.send(message_with_context)
            
            # Default handling
            return self.framework_agent.send(message)
        
        # Fallback if no framework agent
        return f"{self.name} received your message but doesn't know how to respond yet"
    
    async def generate_response_async(self, message: str, conversation_id: str = None) -> str:
        """
        Async version of generate_response.
        Can take an optional conversation_id to maintain context for multiple conversations.
        """
        # Store the conversation ID we're currently responding to
        self.active_conversation_id = conversation_id
        
        # We need to run the synchronous framework agent call in a thread pool
        # to avoid blocking the event loop
        if self.framework_agent:
            # Add conversation context similar to the synchronous version
            if conversation_id:
                # Get the conversation partner for context
                conversation = self.get_conversation(conversation_id)
                if conversation:
                    partner_id = conversation.get_conversation_partner(self.id)
                    if partner_id is not None:
                        # Find the partner's name
                        partner_name = f"Agent_{partner_id}"  # Default
                        for name, agent in self.get_conversation_metadata().get("agent_lookup", {}).items():
                            if hasattr(agent, 'id') and agent.id == partner_id:
                                partner_name = name
                                break
                                
                        # Add conversation context (subtly)
                        message_with_context = f"[Conversation with {partner_name}] {message}"
                        return await asyncio.to_thread(self.framework_agent.send, message_with_context)
            
            # Default handling
            return await asyncio.to_thread(self.framework_agent.send, message)
        
        # Fallback if no framework agent (with short delay to simulate thinking)
        await asyncio.sleep(0.5)
        return f"{self.name} received your message but doesn't know how to respond yet"
        
    def get_conversation_metadata(self) -> Dict:
        """Get metadata about the agent's conversations and environment"""
        # This can be extended to provide additional context
        return {
            "agent_lookup": {},  # This will be populated by the simulation
            "conversation_counts": len(self.conversations)
        }

    def wants_to_move(self) -> bool:
        """Check if agent wants to move (delegated to framework agent if available)"""
        # First check the direct flag (added as a backup)
        if hasattr(self, '_wants_to_move') and self._wants_to_move:
            # Clear the flag after checking it
            self._wants_to_move = False
            logging.info(f"{self.name} wants to move based on direct flag")
            return True
            
        # Then check the framework agent's method
        if self.framework_agent:
            result = self.framework_agent.wants_to_move()
            if result:
                logging.info(f"{self.name} wants to move based on framework agent")
            return result
        
        return False

    def get_state(self) -> Dict:
        # Return current agent state
        state = {
            "id": self.id,
            "name": self.name,
            "history": self.convo_history,
            "state": self.state,
            "thinking": self.thinking,
            "metadata": self.conversation_metadata,
            "active_conversation_id": self.active_conversation_id,
            "conversation_ids": list(self.conversations.keys())
        }
        
        # Add framework agent info if available
        if self.framework_agent:
            state["memory_enabled"] = self.framework_agent.memory_enabled
            state["personality_strength"] = self.framework_agent.personality_strength
            
        return state

    def set_state(self, state: Status):
        # Set the current agent's state
        self.state = state
        
    def update_conversation_metadata(self, key: str, value: Any):
        """Add metadata about conversations for improved history tracking"""
        self.conversation_metadata[key] = value
        
    def get_conversations(self) -> Dict[str, Conversation]:
        """Get all conversations this agent is involved in"""
        return self.conversations
    
    def get_conversation(self, conversation_id: str) -> Optional[Conversation]:
        """Get a specific conversation by ID"""
        return self.conversations.get(conversation_id)
    
    def get_conversation_with_agent(self, agent_id: int) -> Optional[Conversation]:
        """Get the conversation with a specific agent"""
        for conversation in self.conversations.values():
            if conversation.is_between(self.id, agent_id):
                return conversation
        return None
    
    def get_conversation_messages(self, conversation_id: str = None) -> List[Message]:
        """Get messages from a specific conversation, or active conversation if not specified"""
        conversation_id = conversation_id or self.active_conversation_id
        if not conversation_id:
            return []
        
        conversation = self.conversations.get(conversation_id)
        if not conversation:
            return []
            
        return conversation.get_messages()


class Agent(BaseAgent):
    """
    Enhanced agent with memory/personality capabilities that uses framework LLM
    """
    def __init__(
            self,
            agent_id: int,
            name: str,
            provider: str = None,
            api_key: str = None,
            model: str = None,
            system_prompt: str = None,
            memory_enabled: bool = True,
            personality_strength: float = 0.5
    ):
        super().__init__(agent_id, name)
        
        # Create framework agent if provider is specified
        if provider:
            self.framework_agent = create_agent(
                provider=provider,
                name=name,
                api_key=api_key,
                model=model,
                system_prompt=system_prompt or f"You are {name}, a helpful assistant.",
                memory_enabled=memory_enabled,
                personality_strength=personality_strength
            )
        else:
            self.framework_agent = None
            
    def set_memory_enabled(self, enabled: bool):
        """Set memory enabled state"""
        if self.framework_agent:
            self.framework_agent.set_memory_enabled(enabled)
        
    def set_personality_strength(self, strength: float):
        """Set personality strength"""
        if self.framework_agent:
            self.framework_agent.set_personality_strength(strength)

    def get_state(self):
        """Get enhanced state information"""
        state = super().get_state()
        
        # Add memory and personality info
        if self.framework_agent:
            state["Memory Enabled"] = self.framework_agent.memory_enabled
            state["Personality Strength"] = self.framework_agent.personality_strength
        else:
            state["Memory Enabled"] = False
            state["Personality Strength"] = 0.0
            
        return state


class AgentManager:
    """
    Manages multiple agents, message routing, and conversation flow
    """
    def __init__(self, agents: List[BaseAgent]):
        # Store agents by ID
        self.agents = {agent.id: agent for agent in agents}
        # Global conversation registry (conversation_id -> Conversation object)
        self.conversation_registry = {}

    def register_conversation(self, conversation: Conversation) -> str:
        """Register a conversation in the global registry"""
        conversation_id = conversation.conversation_id
        self.conversation_registry[conversation_id] = conversation
        return conversation_id
    
    def get_conversation(self, conversation_id: str) -> Optional[Conversation]:
        """Get a conversation from the registry"""
        return self.conversation_registry.get(conversation_id)
    
    def get_conversations_between(self, agent_id1: int, agent_id2: int) -> List[Conversation]:
        """Get all conversations between two agents"""
        result = []
        for conversation in self.conversation_registry.values():
            if conversation.is_between(agent_id1, agent_id2):
                result.append(conversation)
        return result
        
    def get_agent_conversations(self, agent_id: int) -> List[Conversation]:
        """Get all conversations involving an agent"""
        result = []
        for conversation in self.conversation_registry.values():
            if agent_id in conversation.agent_ids:
                result.append(conversation)
        return result
    
    def get_or_create_conversation(self, agent_id1: int, agent_id2: int, create_new: bool = False) -> Conversation:
        """Get the most recent conversation between two agents, or create a new one"""
        if not create_new:
            # Find the most recent conversation between these agents
            conversations = self.get_conversations_between(agent_id1, agent_id2)
            if conversations:
                # Sort by last_updated time (most recent first)
                conversations.sort(key=lambda c: c.last_updated, reverse=True)
                return conversations[0]
                
        # Create a new conversation
        conversation = Conversation(agent_ids={agent_id1, agent_id2})
        self.register_conversation(conversation)
        
        # Also register with the agents
        if agent_id1 in self.agents:
            self.agents[agent_id1].conversations[conversation.conversation_id] = conversation
        if agent_id2 in self.agents:
            self.agents[agent_id2].conversations[conversation.conversation_id] = conversation
            
        return conversation
    
    def rand_interaction(
            self,
            sender_id: int = None,
            content: str = None,
            conversation_id: str = None
    ) -> Message:
        # Get a random sender if not provided
        if not sender_id:
            sender_id = random.choice(list(self.agents.keys()))

        # Placeholder start message if not provided
        if not content:
            content = "Hello!"

        # Pick a random recipient (excluding sender)
        recipient_id = random.choice(
            [a_id for a_id in self.agents if a_id != sender_id]
        )
        
        # Create message with conversation ID if provided
        msg = Message(sender_id, content, recipient_id, conversation_id)
        
        # If no conversation ID was provided, check if we should create a new one
        if not conversation_id:
            # Get or create conversation between these agents
            conversation = self.get_or_create_conversation(sender_id, recipient_id)
            msg.set_conversation_id(conversation.conversation_id)

        return msg

    def clear_lasts(self):
        for key in self.agents:
            self.agents[key].last_msg = None

    def process_messages(self, msg: Message):
        """
        Runs a conversation given an initial message
        """
        from visualize import visualize  # Import here to avoid circular imports
        
        while True:
            # Deal with response
            recipient = self.agents[msg.recipient_id]
            recipient.set_state(Status.THINKING)
            response = recipient.receive_message(msg)
            recipient.set_state(Status.TALKING)

            # Get these and visualize them
            visualize(list(self.agents.values()))
            self.clear_lasts()

            if not response:
                break

            # Create new message for next iteration
            msg = self.rand_interaction(recipient.id, response)

    async def process_messages_async(self, msg: Message):
        """
        Async version of process_messages that allows for concurrent agent processing
        and provides UI feedback during the thinking state.
        """
        from visualize import visualize  # Import here to avoid circular imports
        
        while True:
            # Deal with response
            recipient = self.agents[msg.recipient_id]
            
            # Process message asynchronously with thinking state indication
            response = await recipient.receive_message_async(msg)
            
            # Visualize the updated state
            visualize(list(self.agents.values()))
            self.clear_lasts()
            
            if not response:
                break
                
            # Create new message for next iteration
            msg = self.rand_interaction(recipient.id, response)
            
    async def run_conversation_async(self, rounds: int = 10):
        """
        Async version of run_conversation that uses the async processing
        """
        for _ in range(rounds):
            # Log a new round
            agent_list = [
                self.agents[key].get_id_pair() for key in self.agents
            ]
            logging.info(f"NEW ASYNC CONVO: {agent_list}")
            
            # Create Random Message
            init_msg = self.rand_interaction()
            
            # Starts Conversation using async method
            await self.process_messages_async(init_msg)

    def run_conversation(self, rounds: int = 10):
        """
        Runs a full conversation simulation for a given number of rounds
        """
        for _ in range(rounds):
            # Log a new round
            agent_list = [
                self.agents[key].get_id_pair() for key in self.agents
            ]
            logging.info(f"NEW CONVO: {agent_list}")

            # Create Random Message
            init_msg = self.rand_interaction()

            # Starts Conversation
            self.process_messages(init_msg)

    def get_agent_states(self) -> Dict[int, Dict]:
        # Retrieve the state of all agents for UI or debugging
        return {aid: agent.get_state() for aid, agent in self.agents.items()}
    
    def get_agent_movement_states(self) -> Dict[int, bool]:
        """Check which agents want to move"""
        return {aid: agent.wants_to_move() for aid, agent in self.agents.items()}


def create_agent_with_llm(
    agent_id: int,
    name: str,
    provider: str,
    api_key: str,
    model: str = None,
    system_prompt: str = None,
    memory_enabled: bool = True,
    personality_strength: float = 0.5
) -> Agent:
    """
    Factory function to create an Agent with integrated LLM capabilities
    """
    return Agent(
        agent_id=agent_id,
        name=name,
        provider=provider,
        api_key=api_key,
        model=model,
        system_prompt=system_prompt,
        memory_enabled=memory_enabled,
        personality_strength=personality_strength
    )


if __name__ == "__main__":
    # Example usage
    from dotenv import load_dotenv
    load_dotenv()
    
    openai_key = os.getenv("OPENAI_API_KEY")
    
    # Create agents with LLM capabilities
    agent1 = create_agent_with_llm(
        agent_id=1,
        name="Alice",
        provider="openai",
        api_key=openai_key,
        model="gpt-3.5-turbo",
        system_prompt="You are Alice, a friendly assistant. Keep responses short."
    )
    
    agent2 = create_agent_with_llm(
        agent_id=2,
        name="Bob",
        provider="openai",
        api_key=openai_key,
        model="gpt-3.5-turbo",
        system_prompt="You are Bob, a technical expert. Keep responses short."
    )
    
    # Simple agent without LLM (for testing)
    agent3 = BaseAgent(3, "Charlie")
    
    # Create manager and run a test conversation
    manager = AgentManager([agent1, agent2, agent3])
    manager.run_conversation(1)
    
    # Print agent states for debugging
    for agent_id, state in manager.get_agent_states().items():
        print(f"Agent {agent_id}:")
        for key, value in state.items():
            if key != "history":  # Skip printing the full history
                print(f"  {key}: {value}")
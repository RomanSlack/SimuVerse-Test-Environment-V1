from enum import Enum
import random
import logging
import os
import asyncio
from typing import List, Dict, Optional, Union, Tuple, Coroutine, Any
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
    def __init__(self, sender_id: int, content: str, recipient_id: int = None):
        self.sender_id = sender_id
        self.content = content
        self.recipient_id = recipient_id

    def __str__(self):
        # Prints the message in readable format
        return f"{self.sender_id} -> {self.recipient_id}: {self.content}"

    def set_recipient(self, id: int):
        # Sets the reciever of a message
        self.recipient_id = id


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

    def receive_message(self, msg: Message):
        # Store received message
        self.convo_history.append(msg)
        self.last_msg = msg
        # Logs message
        logging.info(str(msg))

        # Generate a response
        return self.generate_response(msg.content)
    
    async def receive_message_async(self, msg: Message) -> Optional[str]:
        """Async version of receive_message"""
        # Store received message
        self.convo_history.append(msg)
        self.last_msg = msg
        # Logs message
        logging.info(str(msg))

        # Set thinking state for UI
        self.thinking = True
        self.set_state(Status.THINKING)
        
        # Generate a response asynchronously
        response = await self.generate_response_async(msg.content)
        
        # Reset thinking state
        self.thinking = False
        self.set_state(Status.TALKING)
        
        return response

    def generate_response(self, message: str) -> str:
        """Base response generation"""
        if self.framework_agent:
            # Use the framework agent if available
            return self.framework_agent.send(message)
        
        # Fallback if no framework agent
        return f"{self.name} received your message but doesn't know how to respond yet"
    
    async def generate_response_async(self, message: str) -> str:
        """Async version of generate_response"""
        # We need to run the synchronous framework agent call in a thread pool
        # to avoid blocking the event loop
        if self.framework_agent:
            return await asyncio.to_thread(self.framework_agent.send, message)
        
        # Fallback if no framework agent (with short delay to simulate thinking)
        await asyncio.sleep(0.5)
        return f"{self.name} received your message but doesn't know how to respond yet"

    def wants_to_move(self) -> bool:
        """Check if agent wants to move (delegated to framework agent if available)"""
        if self.framework_agent:
            return self.framework_agent.wants_to_move()
        return False

    def get_state(self) -> Dict:
        # Return current agent state
        state = {
            "id": self.id,
            "name": self.name,
            "history": self.convo_history,
            "state": self.state,
            "thinking": self.thinking,
            "metadata": self.conversation_metadata
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

    def rand_interaction(
            self,
            sender_id: int = None,
            content: str = None
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

        return Message(sender_id, content, recipient_id)

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
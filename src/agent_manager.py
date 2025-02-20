import random
import logging
from collections import deque
from typing import List, Dict, Optional
from logging.handlers import RotatingFileHandler

# Configuring the logging
log_handler = RotatingFileHandler("logs/conversations.log", maxBytes=25_000_000, backupCount=3)
log_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(log_handler)

class BaseAgent:
    # Basic agent with identifiers

    def __init__(self, agent_id: int, name: str):
        self.id = agent_id
        self.name = name
        self.conversation_history = []
        self.state = {}

    def __repr__(self):
        return f"{self.id}: {self.name}\n\t{self.conversation_history}\n\t{self.state}\n\t"
    
    def receive_message(self, message: str):
        # Store received messages
        self.conversation_history.append(message)
    
    def generate_response(self, message: str) -> str:
        # Response generation
        return f"{self.name} received: {message}"       # TODO actually implement later

    def get_state(self) -> Dict:
        # Return current agent state
        return {"id": self.id, "name": self.name, "history": self.conversation_history, "state": self.state}
    
    def set_state(self, state: dict):
        # Set the current agent's state
        self.state = state

class Agent(BaseAgent):
    # Actual agent with memory/personality capabilities

    def __init__(self, agent_id: int, name: str, memory_enabled=False, personality_enabled=False):
        super().__init__(agent_id, name)
        self.memory_enabled = memory_enabled
        self.personality_enabled = personality_enabled
        self.memory = [] if memory_enabled else None                     # TODO implement later
        self.personality = {} if personality_enabled else None    # TODO implement later
    
    def store_memory(self, message: str):
        # Store message in memory if enabled
        if self.memory_enabled:
            self.memory.append(message)
    
    def generate_response(self, message: str) -> str:
        # Generate response with memory and personality influence
        self.receive_message(message)
        self.store_memory(message)
        
        base_response = super().generate_response(message)
        
        # TODO implement modifications based on personality/memory
        
        return base_response[:500]      # limit size of response

    def get_state(self):
        state = super().get_state()
        new_state = {"Memory Enabled": self.memory_enabled, "Memory": self.memory, "Personality Enabled": self.personality_enabled, "Personality": self.personality}
        for key in new_state:
            state[key] = new_state[key]
        return state
    
class AgentManager:
    # Manages multiple agents, message routing, and conversation flow

    def __init__(self, agents: List[Agent]):
        self.agents = {agent.id: agent for agent in agents}  # Store agents by ID
        self.message_queue = deque()  # FIFO queue for processing messages
    
    def route_message(self, sender_id: int, message: str):
        # Send a message from one agent to another and queue a response
        if sender_id not in self.agents:
            logging.warning(f"Message from unknown agent {sender_id}")
            return
        
        # Pick a random recipient (excluding sender)
        recipient_id = random.choice([a_id for a_id in self.agents if a_id != sender_id])
        
        logging.info(f"{self.agents[sender_id].name} --> {self.agents[recipient_id].name}: {message}")
        
        # Enqueue the message
        self.message_queue.append((sender_id, recipient_id, message))

    def process_messages(self):
        # Process all messages in the queue, ensuring natural conversation flow
        while self.message_queue:
            queue_size = len(self.message_queue)  # Snapshot of current queue
            for _ in range(queue_size):  # Prevent infinite looping
                sender_id, recipient_id, message = self.message_queue.popleft()
                recipient = self.agents[recipient_id]

                # Generate response
                response = recipient.generate_response(message)

                logging.info(f"{recipient.name} (Response): {response}")

                if response and random.random() < 0.3:  # 30% chance of continuing
                    self.message_queue.append((recipient_id, sender_id, response))

    def run_conversation(self, rounds: int = 10):
        # Runs a full conversation simulation for a given number of rounds
        for _ in range(rounds):
            sender_id = random.choice(list(self.agents.keys()))
            initial_message = "Hello!"  # Placeholder start message
            print(_)
            self.route_message(sender_id, initial_message)
            self.process_messages()

    def get_agent_states(self) -> Dict[int, Dict]:
        # Retrieve the state of all agents for UI or debugging
        return {aid: agent.get_state() for aid, agent in self.agents.items()}
    
if __name__ == "__main__":
    agents = [
        Agent(1, "Alice", memory_enabled=False, personality_enabled=False),
        Agent(2, "Bob", memory_enabled=False, personality_enabled=False),
        Agent(3, "Charlie", memory_enabled=False, personality_enabled=False),
    ]
    
    manager = AgentManager(agents)
    manager.run_conversation(5)

    # Print agent states for debugging
    for state in manager.get_agent_states().values():
        print(state)

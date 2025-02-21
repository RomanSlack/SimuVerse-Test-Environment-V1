import random
import logging
from typing import List, Dict
from logging.handlers import RotatingFileHandler

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
    # Basic agent with identifiers

    def __init__(self, agent_id: int, name: str):
        self.id = agent_id
        self.name = name
        self.convo_history = []
        self.state = {}

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
        # Logs message
        logging.info(str(msg))

        # Respond with 50% chance
        if random.random() < 0.5:
            return self.generate_response()
        return None

    def generate_response(self) -> str:
        # Response generation
        return "Responded"       # TODO actually implement later

    def get_state(self) -> Dict:
        # Return current agent state
        return {
            "id": self.id,
            "name": self.name,
            "history": self.convo_history,
            "state": self.state
        }

    def set_state(self, state: dict):
        # Set the current agent's state
        self.state = state


class Agent(BaseAgent):
    # Actual agent with memory/personality capabilities

    def __init__(
            self,
            agent_id: int,
            name: str,
            memory_enabled=False,
            personality_enabled=False
    ):
        super().__init__(agent_id, name)
        self.memory_enabled = memory_enabled
        self.personality_enabled = personality_enabled

        # TODO implement later
        self.memory = [] if memory_enabled else None

        # TODO implement later
        self.personality = {} if personality_enabled else None

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
        new_state = {
            "Memory Enabled": self.memory_enabled,
            "Memory": self.memory,
            "Personality Enabled": self.personality_enabled,
            "Personality": self.personality
        }
        for key in new_state:
            state[key] = new_state[key]
        return state


class AgentManager:
    # Manages multiple agents, message routing, and conversation flow

    def __init__(self, agents: List[Agent]):
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
            initial_message = "Hello!"

        # Pick a random recipient (excluding sender)
        recipient_id = random.choice(
            [a_id for a_id in self.agents if a_id != sender_id]
        )

        return Message(sender_id, initial_message, recipient_id)

    def process_messages(self, msg: Message):
        """
        Runs a conversation given an initial message
        """
        while True:
            # Deal with response
            recipient = self.agents[msg.recipient_id]
            response = recipient.receive_message(msg)
            if not response:
                break

            # Create new message for next iteration
            msg = self.rand_interaction(recipient.id, response)

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


if __name__ == "__main__":
    agents = [
        BaseAgent(1, "Alice"),
        BaseAgent(2, "Bob"),
        BaseAgent(3, "Charlie"),
    ]

    manager = AgentManager(agents)
    manager.run_conversation(1)

    # Print agent states for debugging
    print(manager.get_agent_states())

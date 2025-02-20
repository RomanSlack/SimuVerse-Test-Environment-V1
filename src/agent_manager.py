
class BaseAgent:
    def __init__(self):
        self.state = {}
        self.last_message = None
        self.conversation_history = []

    def receive_message(self, msg):
        # Receive a message from the user
        self.conversation_history.append(msg)

    def generate_response(self):
        # Generate a response based on the current state
        pass

    def get_state(self):
        # Return the current state of the agent
        return self.state

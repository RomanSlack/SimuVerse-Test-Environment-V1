# Simuverse: Multi-Agent Testing Environment – Project Outline

## 1. Project Overview and Goals

### Purpose
Build a standalone Python-based testing environment that allows up to 10+ AI agents (powered by LLMs) to converse, simulate movement, and integrate optional features like memory and personality.

### Key Objectives
- Rapidly prototype and debug multi-agent interactions without requiring the full Unity environment
- Provide a plug-and-play architecture for adding or removing modules (e.g., memory, personality) at will
- Implement an innovative UI that supports simultaneous agent conversations in a clear, trackable format
- Ensure comprehensive logging and debugging capabilities

## 2. High-Level Architecture

### Core Python Application
- Acts as the controller for all agents
- Houses logic for conversation flow, message routing, and state management
- Serves as the integration point for various modules (memory, personality, etc.)

### Agent Modules
- Base Agent: Minimal agent powered by an LLM (no memory, no personality)
- Memory-Enhanced Agent: Integrates with a vector database for short-term and/or long-term memory storage and retrieval
- Personality-Enhanced Agent: Includes a set of personality traits or parameterized profile that influences its conversational outputs
- Agents can be instantiated in any combination of these enhancements

### UI / Visualization Layer
- A custom interface (2D or minimal 3D if efficient) showing real-time agent dialogues and simulated positioning
- Should handle up to 10+ agents simultaneously in a concise manner

## 3. Step-by-Step Development Plan

### 3.1 Environment Setup

#### Create Project Structure
- src/ folder containing the main application code (e.g., main.py, agent_manager.py)
- modules/ folder for optional components: memory, personality, etc.
- ui/ folder for front-end or console-based UI code
- logs/ folder for storing log files, conversation transcripts, error logs, etc.

#### Dependency Management
- Use requirements.txt or a virtual environment (e.g., conda or venv)
- Ensure standard LLM integration libraries (e.g., openai, or other relevant libraries) are included

### 3.2 Agent Management and Core Logic

#### Agent Definition
- Define a BaseAgent class with methods like receive_message(msg), generate_response(), and get_state()
- Store minimal internal state (e.g., last message, conversation history)

#### Agent Manager / Orchestrator
- A central class (e.g., AgentManager) that tracks all agents, routes messages, and updates states
- Implements round-robin or event-based conversation flow so that messages from one agent can trigger responses in others

#### Conversation Flow
- Implement a conversation loop that allows for synchronous or asynchronous message passing
- Provide a mechanism to pause or step through each round of conversation for debugging

### 3.3 Memory System Integration

#### Memory Module
- A specialized class or set of classes (e.g., MemoryManager) that interfaces with a vector database (like FAISS, Pinecone, Chroma, etc.)
- Provides functions like save_embedding(agent_id, text), retrieve_relevant(agent_id, query)

#### Agent Hooks
Memory-Enabled Agent inherits from BaseAgent and overrides generate_response() or additional methods to consult memory:
- Store new conversation pieces as embeddings
- Retrieve relevant past context before generating a response

#### Configurable Usage
The environment (via the Agent Manager) can enable or disable the memory feature per agent or globally.

### 3.4 Personality System Integration

#### Personality Data
- Define a lightweight data structure for personality traits (e.g., traits = {"optimism": 0.8, "patience": 0.2, ...}) or a more narrative-based profile

#### Personality-Enhanced Agent
- Inherits from BaseAgent, modifies the prompt or internal logic to reflect personality traits
- Could alter temperature or style parameters in the LLM request or use a personality text prompt preamble

#### Optional Usage
As with memory, the personality module can be activated or deactivated at agent instantiation.

## 4. Innovative UI / Visualization

### Conceptual Overview
- Instead of a simple chat window, design an agent grid or network view where each agent is represented by a node or avatar
- Each node displays agent name, status (talking, thinking, idle), and recent message

### Conversation Flow Visualization
- Conversation Threads: Draw speech balloons or a small timeline branching from each agent's node
- Use color-coded lines between agents to show who is addressing whom
- Optionally show a scrollable timeline listing each utterance chronologically

### Movement Simulation (Optional/Low-Fidelity)
- Represent each agent as a simple icon on a 2D plane (or minimal 3D if performance allows)
- Agents move to "conversation clusters" or "rooms" so it appears they are physically navigating a space (though no real movement logic is needed)
- Each agent's position can update in real-time to simulate them "moving" or "grouping" for certain discussions

### Implementation Options
- Tkinter or PyQt for a simple but interactive 2D node-based UI
- Web-based UI (using Flask/Streamlit) displaying a graph layout (D3.js or similar) for real-time agent positions and speech balloons

### User Interaction
- Start/Stop Simulation buttons
- Agent Control Panel to enable/disable memory or personality on the fly
- Step-by-Step mode to iterate manually through conversation turns

## 5. Logging and Debugging

### Logging Strategy
- Conversation Logs: Every message stored in a human-readable format (timestamp, agent ID, message content)
- Error Logs: Any exceptions, LLM errors, or system anomalies
- Performance Metrics (optional): Response times, memory retrieval times, etc.

### Debug Tools
- Debug Console in the UI to display real-time logs
- Option to export conversation transcripts and memory usage data

### Configuration
- Logging levels (debug, info, warning, error)
- Central logging config in a dedicated config file (logging.conf or similar)

## 6. Testing & Validation

### Unit Tests
- For BaseAgent, memory retrieval, personality injection, etc.
- Validate that toggling features on/off works as expected

### Integration Tests
- Multi-agent conversation scenario with a script controlling the environment
- Verify logs, UI displays, memory retrieval accuracy, etc.

### Performance Tests
- Test concurrency with 10+ agents to ensure stable performance
- Evaluate LLM rate limits and fallback mechanisms

## 7. Implementation Roadmap

### Milestone 1: Basic Multi-Agent Conversation
- Implement BaseAgent, AgentManager, a simple console-based UI
- Validate conversation flow with 2–3 agents

### Milestone 2: UI Prototype
- Develop the innovative node-based UI (Tkinter, PyQt, or web-based)
- Display real-time interactions in a grid/network format

### Milestone 3: Memory and Personality Modules
- Integrate vector database for memory (optional for each agent)
- Add personality trait system (optional for each agent)
- Ensure easy toggling of each feature

### Milestone 4: Logging & Debug Enhancements
- Implement robust logging (conversation transcripts, error logs, etc.)
- Create debug tools (step-by-step mode, console)

### Milestone 5: Movement Simulation (If Required)
- Implement simple 2D or minimal 3D layout for agent "movement"
- Visualize movement states in the UI

### Milestone 6: Final Integration & Testing
- Conduct end-to-end tests with all features enabled
- Optimize performance for 10+ agents
- Gather feedback, refine UI

## 8. Next Steps
- Assign Tasks: Distribute each milestone to respective team members (UI/UX developer, memory module developer, personality lead, etc.)
- Define Timelines: Estimate time per milestone and set up a sprint schedule
- Gather Feedback: Continual reviews after each milestone to ensure alignment with Unity environment goals
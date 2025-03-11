# SimuVerse: Multi-Agent Interactive Simulation Environment

SimuVerse is a dynamic multi-agent simulation framework that enables AI agents to interact in a social environment, move around autonomously, and engage in conversations. The project creates a visual grid-based interface where different LLM-powered agents can form connections, communicate, and navigate their social world.

## Features

- **Interactive Agent Grid**: Visualize agents and their connections in real-time
- **Dynamic Movement**: Agents autonomously move to meet new conversation partners
- **Multi-LLM Support**: Use OpenAI, Claude, Hugging Face, and other LLM providers
- **Conversation Tracking**: Monitor and visualize agent interactions
- **Agent Personality Settings**: Adjust memory and personality strength parameters
- **Movement Intent Detection**: Agents can explicitly request to move using commands

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/SimuVerse.git
cd SimuVerse
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install the required packages:
```bash
pip install -r requirements.txt
```

4. Create a `.env` file with your API keys:
```
OPENAI_API_KEY=your_openai_api_key_here
CLAUDE_API_KEY=your_claude_api_key_here
```

## Running the Simulation

To start the web-based simulation interface:

```bash
python simulation_grid_test.py
```

Then open your browser and navigate to `http://127.0.0.1:8050/` to view the simulation.

## How It Works

### Agent Framework

The core of SimuVerse is a flexible multi-agent framework that supports various LLM providers:

- **BaseLLM Interface**: Abstract class for implementing different LLM backends
- **Agent Class**: Manages agent state, conversation history, and movement requests
- **MultiAgentFramework**: Container for managing multiple agents and their interactions

### Simulation Environment

The simulation runs in a grid-based environment where:

1. Agents are positioned on a 2D grid
2. Connections form between agents based on proximity
3. Connected agents exchange messages
4. After 2-3 conversation rounds, agents may decide to move
5. Movement can be triggered automatically or by agent request using `[MOVE]` command

### User Interface

The web interface allows you to:

- Visualize agent positions and connections
- Track conversation logs between agents
- Monitor movement probabilities and intentions
- Step through the simulation
- Manually reposition agents
- Adjust agent memory and personality settings

## Agent Movement Mechanisms

Agents can move in two ways:

1. **Autonomous Movement**: After 2-3 rounds of conversation, agents build up a probability of moving based on their personality and conversation duration.

2. **Explicit Movement Requests**: Agents can include `[MOVE]` in their responses to explicitly request movement to meet someone new.

## Next Steps: Taking SimuVerse to the Next Level

Here are some exciting directions for future development:

### 1. Enhanced Environment

- **Multiple Rooms/Zones**: Create distinct areas with different conversation topics or themes
- **Environmental Factors**: Introduce noise, crowding, or other factors that affect communication
- **Resource Management**: Add resources agents must gather, share, or compete for
- **Day/Night Cycles**: Implement time progression affecting agent behavior
- **Weather/Conditions**: Environmental factors that influence mood or behavior

### 2. Advanced Agent Capabilities

- **Goal-Directed Behavior**: Allow agents to form and pursue goals
- **Learning & Memory**: Implement more sophisticated memory models and learning from interactions
- **Emotional States**: Track agent emotions and have them affect interactions
- **Relationship Development**: Track relationship status between agent pairs
- **Tool Usage**: Allow agents to use various tools in the environment
- **Advanced Movement Commands**: Let agents specify direction, destination, or companions

### 3. Improved Visualization & Analysis

- **Network Analysis**: Add social network metrics and visualization
- **Conversation Heatmaps**: Show areas with most active conversations
- **Time-series Analysis**: Track and visualize how relationships evolve over time
- **Agent Journey Maps**: Trace paths of agents through social space
- **3D Visualization**: Upgrade to a 3D visualization for more immersive experience

### 4. System Architecture Improvements

- **Scalability**: Optimize for hundreds or thousands of agents
- **Distributed Computation**: Allow simulation to run across multiple machines
- **Persistent Storage**: Save simulation states to database for long-running experiments
- **API Integration**: Connect with external systems and data sources
- **Containerization**: Package the system in Docker for easy deployment

### 5. User Experience Enhancements

- **Interactive Timeline**: Scroll through simulation history
- **Agent Configuration UI**: More detailed controls for agent creation and modification
- **Scenario Builder**: Create pre-defined scenarios and experiments
- **Mobile-Friendly Design**: Better support for mobile devices
- **Real-time Intervention**: Allow users to send messages or commands to agents during simulation

### 6. Research Applications

- **Sociological Modeling**: Study emergent social behaviors
- **Communication Patterns**: Analyze how information spreads through agent networks
- **Group Formation**: Study how cliques, communities, and hierarchies form
- **Cultural Evolution**: Model how cultural norms develop and spread
- **Misinformation Studies**: Track how false information propagates through networks

### 7. Integration with Unity or Game Engines

- **Upgrade to Unity**: Move the visualization layer to Unity for more sophisticated rendering
- **VR Support**: Enable virtual reality viewing of the simulation
- **Interactive Objects**: Add interactive objects and environments within the simulation
- **Physics-Based Movement**: Implement realistic movement with collision detection
- **Advanced Animation**: Add character models and animations for more realistic representation

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- This project uses [Dash](https://dash.plotly.com/) for the web interface
- Agent networks are visualized using [Cytoscape.js](https://js.cytoscape.org/)
- LLM capabilities are provided by OpenAI and Anthropic APIs
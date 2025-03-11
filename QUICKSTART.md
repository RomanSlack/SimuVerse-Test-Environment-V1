# SimuVerse Quick Start Guide

This guide will help you quickly set up and run the SimuVerse multi-agent simulation environment.

## Prerequisites

- Python 3.8+ installed on your system
- OpenAI API key (for GPT models)
- Claude API key (for Claude models)
- Git (for cloning the repository)

## 5-Minute Setup

### 1. Clone and Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/SimuVerse.git
cd SimuVerse

# Create a virtual environment
python -m venv venv

# Activate the virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure API Keys

Create a file named `.env` in the project root with your API keys:

```
OPENAI_API_KEY=your_openai_api_key_here
CLAUDE_API_KEY=your_claude_api_key_here
```

### 3. Run the Simulation

```bash
python simulation_grid_test.py
```

Open your web browser and navigate to `http://127.0.0.1:8050/` to view the simulation interface.

## Basic Usage

### Simulation Controls

- **Step Simulation Button**: Advances the simulation by one step
- **Agent Display**: Shows agent positions and connections
- **Conversation Logs**: Click on connections to view conversation history
- **Agent Settings**: Adjust memory and personality strength for each agent

### Agent Interaction

1. **Moving Agents**: Drag agent nodes to reposition them
2. **Forming Connections**: Agents connect to their closest neighbor
3. **Conversations**: Agents communicate with connected agents
4. **Movement**: After 2-3 rounds of conversation, agents may move to meet new partners

## Key Features

### Autonomous Movement

Agents will automatically consider moving after 2-3 rounds of conversation with the same partner. They weigh:
- Their personality (movement probability)
- How long they've been talking to the same person
- A random chance element

### Agent Movement Commands

Agents can explicitly request to move by including `[MOVE]` in their responses. This simulates an agent's desire to change conversation partners.

### Visual Indicators

- **Red Dashed Border**: Agent is likely to move soon
- **Yellow Dashed Border**: Agent may move soon
- **Blue Border**: Agent has just moved or joined a conversation
- **Red Dashed Lines**: New connections between agents
- **Walking Icon**: Agent has explicitly requested to move

## Troubleshooting

### API Connection Issues

If you see errors related to API connections:
1. Check that your API keys in the `.env` file are correct
2. Ensure you have internet connectivity
3. Verify you have sufficient API credits/quota

### Display Problems

If the web interface doesn't load properly:
1. Check that port 8050 is available (not used by another application)
2. Try refreshing the browser
3. Check browser console for any JavaScript errors

### Agent Issues

If agents aren't responding as expected:
1. Check the API key for that specific agent's provider (OpenAI or Claude)
2. Look at the memory settings to ensure they're enabled if needed
3. Adjust personality strength to modify behavior

## Next Steps

After getting familiar with the basic simulation:

1. Try modifying agent system prompts in `simulation_grid_test.py` to create different personalities
2. Experiment with different movement probabilities and conversation patterns
3. Check the `IMPROVEMENTS.md` file for planned enhancements
4. Read the full `README.md` for detailed documentation

## Need Help?

For more detailed information, check the following resources:
- Full documentation in the `docs/` directory
- Movement system details in `docs/MOVEMENT.md`
- API documentation for each module in the source code
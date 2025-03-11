# SimuVerse Agent Movement System

This document explains the agent movement capabilities in SimuVerse, including both autonomous and explicit movement mechanisms.

## Movement Overview

In SimuVerse, agents can move around the 2D grid to simulate a social environment where individuals navigate between different conversations and social groups. Movement is a key feature that enables dynamic social interactions and prevents agents from getting "stuck" in the same conversations.

## Movement Mechanisms

SimuVerse supports two primary movement mechanisms:

### 1. Autonomous Movement

Agents will automatically consider moving after engaging in conversation with the same partner for 2-3 rounds. This simulates natural social behavior where people tend to circulate in social settings.

#### How Autonomous Movement Works:

1. **Conversation Tracking**: The system tracks how many rounds each agent pair has conversed
2. **Movement Cooldown**: As conversation rounds increase, a "movement cooldown" counter increases
3. **Probability Calculation**: Movement probability is calculated based on:
   - Agent's base movement probability (personality trait)
   - Current cooldown value (longer conversations = higher movement probability)
4. **Random Factor**: A random roll determines if the agent moves based on the calculated probability
5. **Movement Execution**: If triggered, agent moves in a random direction within grid bounds

#### Movement Probability Formula:

```
probability = min(0.9, base_probability * (1 + 0.2 * cooldown))
```

Where:
- `base_probability` typically ranges from 0.3 to 0.7 depending on agent personality
- `cooldown` increases after 2-3 rounds of conversation with the same partner
- Maximum probability is capped at 90%

### 2. Explicit Movement Requests

Agents can explicitly decide to move by including a special command in their responses. This simulates deliberate decisions to leave a conversation.

#### How Explicit Movement Works:

1. **Movement Command**: Agents include `[MOVE]` anywhere in their response text
2. **Command Detection**: The system detects the command and flags the agent for movement
3. **Command Removal**: The `[MOVE]` tag is removed from the displayed response
4. **Movement Execution**: The agent moves to a new position in the next simulation step
5. **Notification**: A system message informs the agent and UI about the movement

## Visual Indicators

The SimuVerse UI provides several visual cues about agent movement:

- **Red Dashed Border**: Agent is highly likely to move soon (>70% probability)
- **Yellow Dashed Border**: Agent may move soon (40-70% probability)
- **Blue Border**: Agent has just moved or joined a new conversation
- **Walking Icon**: Agent has explicitly requested to move
- **Red Dashed Edge Line**: New connection between agents

## Adding Movement to Agent Behaviors

To leverage the movement system in agent design:

1. **In System Prompts**: Include instructions about when the agent should consider moving
2. **Personality-Based Movement**: Adjust movement probabilities based on agent personality traits
3. **Conversational Triggers**: Design agents to request movement after certain conversation patterns
4. **Movement Commands**: Use the `[MOVE]` command explicitly in agent responses

## Implementation Details

The movement system is primarily implemented in:
- `modules/framework.py`: Agent class with movement request detection
- `simulation_grid_test.py`: Movement logic, probability calculation, and visualization

## Future Enhancements

Planned enhancements to the movement system include:
- Directional movement (agents can specify where to move)
- Group movement (agents can move together)
- Attraction/repulsion factors between certain agent types
- Environment-influenced movement (areas that attract or repel agents)
- Path visualization (showing agent movement history)
import logging
import streamlit as st
# import os
import math
import dash
from dash import html, dcc, Input, Output, State
import dash_cytoscape as cyto
from dash import dash_table
from dash import ALL
import dash_bootstrap_components as dbc
from dotenv import load_dotenv

# from modules.framework import create_agent
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.agent_manager import create_agent_with_llm, Agent, AgentManager, Status, Message

# -------------------------
# Load API Keys and Create Agents
# -------------------------

load_dotenv(override=True)
openai_api_key = os.getenv("OPENAI_API_KEY")
claude_api_key = os.getenv("CLAUDE_API_KEY")

if not openai_api_key or not claude_api_key:
    st.error("One or more API keys not found in environment variables.")
    st.stop()

# Create agents using the integrated AgentManager framework
ava = create_agent_with_llm(
    agent_id=1,
    name="Ava",
    provider="openai",
    api_key=openai_api_key,
    model="gpt-4o-mini",
    system_prompt=(
        "You are Ava, a friendly 20 yr old female college student in a social simulation. "
        "Respond naturally in a conversational tone and limit your reply to no more than 2 sentences. "
        "After talking to the same person for 2-3 rounds, you prefer to move and meet someone new. "
        "You're curious and enjoy meeting different people. "
        "When you want to move to meet someone new, include the exact text [MOVE] somewhere in your response. "
        "This will cause you to physically move in the simulation to meet someone else."
    ),
    memory_enabled=True,
    personality_strength=0.7
)

miles = create_agent_with_llm(
    agent_id=2,
    name="Miles",
    provider="claude",
    api_key=claude_api_key,
    model="claude-3-5-haiku-20241022",
    system_prompt=(
        "You are Miles, an engaging 20 yr old male computer scientist in a social simulation. "
        "Respond concisely and in a human-like manner in no more than 2 sentences. "
        "After talking to the same person for 2-3 exchanges, you prefer to move around and meet new people. "
        "You're outgoing and enjoy diverse conversations. "
        "When you want to move to meet someone new, include the exact text [MOVE] somewhere in your response. "
        "This will cause you to physically move in the simulation to meet someone else."
    ),
    memory_enabled=True,
    personality_strength=0.5
)

lila = create_agent_with_llm(
    agent_id=3,
    name="Lila",
    provider="claude",
    api_key=claude_api_key,
    model="claude-3-5-haiku-20241022",
    system_prompt=(
        "You are Lila, a 20 yr old female soldier from South Korea in a social simulation. "
        "Respond concisely and in a human-like manner in no more than 2 sentences. "
        "After talking to the same person for 2-3 exchanges, you like to move to a new location and meet different people. "
        "You're disciplined but enjoy socializing with various individuals. "
        "When you want to move to meet someone new, include the exact text [MOVE] somewhere in your response. "
        "This will cause you to physically move in the simulation to meet someone else."
    ),
    memory_enabled=True,
    personality_strength=0.9
)

theo = create_agent_with_llm(
    agent_id=4,
    name="Theo",
    provider="openai",
    api_key=openai_api_key,
    model="gpt-4o-mini",
    system_prompt=(
        "You are Theo, a 20 yr old male electrician working at NASA in a social simulation. "
        "Respond naturally in a conversational tone and limit your reply to no more than 2 sentences. "
        "After talking to the same person for 2-3 exchanges, you tend to move to a different area to meet new people. "
        "You're technically minded but enjoy diverse social interactions. "
        "When you want to move to meet someone new, include the exact text [MOVE] somewhere in your response. "
        "This will cause you to physically move in the simulation to meet someone else."
    ),
    memory_enabled=True,
    personality_strength=0.3
)

# Create the AgentManager
agents = [ava, miles, lila, theo]
agent_manager = AgentManager(agents)

# Dictionary mapping agent names to agent objects for lookup
agent_lookup = {
    "Ava": ava,
    "Miles": miles,
    "Lila": lila,
    "Theo": theo
}

# Share the agent lookup with each agent so they can identify conversation partners
for name, agent in agent_lookup.items():
    agent.update_conversation_metadata("agent_lookup", agent_lookup)

# -------------------------
# Simulation Data Structures
# -------------------------
# Initial agent positions (could be randomized or pre-set)
agent_positions = {
    "Ava": {"x": 100, "y": 200},
    "Miles": {"x": 300, "y": 200},
    "Lila": {"x": 100, "y": 400},
    "Theo": {"x": 300, "y": 400}
}

# For tracking conversation logs based on conversation ID
conversation_logs = {name: [] for name in agent_lookup.keys()}
# Track conversation IDs for each connection pair
connection_conversation_ids = {}

# Track conversation rounds between agents
conversation_rounds = {}  # Format: {(source, target): count}

# Track movement state
agent_movement_cooldown = {name: 0 for name in agent_lookup.keys()}  # Countdown until agent considers moving
agent_movement_probability = {name: 0.7 for name in agent_lookup.keys()}  # Base probability of movement
grid_bounds = {"min_x": 50, "max_x": 550, "min_y": 50, "max_y": 550}  # Grid boundaries
movement_distance = 75  # How far agents move in one step


def compute_edges(positions):
    """
    Compute bidirectional edges between agents.
    Allows multiple 2-way connections between agents with proper arrow rendering.
    """
    edges = []
    names = list(positions.keys())
    
    # Get previous connections to detect changes
    previous_connections = getattr(simulation_step, 'previous_connections', {}) if 'simulation_step' in globals() else {}
    
    # Maximum distance for automatic connections - adjust this value to control connection density
    # Higher value = more connections, lower value = fewer connections
    max_connection_distance = 200  # This is in graph units
    
    # Create proper bidirectional edges for each agent pair
    for i, name1 in enumerate(names):
        for j, name2 in enumerate(names[i+1:], i+1):  # Only process each pair once
            if name1 == name2:
                continue  # Skip self-connections
                
            # Calculate distance between agents
            dx = positions[name1]["x"] - positions[name2]["x"]
            dy = positions[name1]["y"] - positions[name2]["y"]
            dist = math.sqrt(dx * dx + dy * dy)
            
            # Connect if within maximum distance
            if dist <= max_connection_distance:
                # Get conversation ID for this pair
                source_agent = agent_lookup[name1]
                target_agent = agent_lookup[name2]
                
                # Check if this is a new connection
                previous_connections_1 = previous_connections.get(name1, set()) if isinstance(previous_connections.get(name1, None), set) else set()
                previous_connections_2 = previous_connections.get(name2, set()) if isinstance(previous_connections.get(name2, None), set) else set()
                
                is_new_connection = name2 not in previous_connections_1 or name1 not in previous_connections_2
                
                # Add visual class to mark new connections
                edge_class = "new-connection" if is_new_connection else ""
                
                # Find or create conversation ID for this pair
                edge_pair = (name1, name2)
                reverse_edge_pair = (name2, name1)
                
                # Use the same conversation ID for both directions
                if edge_pair in connection_conversation_ids:
                    conversation_id = connection_conversation_ids[edge_pair]
                elif reverse_edge_pair in connection_conversation_ids:
                    conversation_id = connection_conversation_ids[reverse_edge_pair]
                else:
                    # Try to get existing conversation between these agents
                    conversation = source_agent.get_conversation_with_agent(target_agent.id)
                    
                    if conversation:
                        conversation_id = conversation.conversation_id
                    else:
                        # Create a new conversation when agents connect
                        if is_new_connection and hasattr(agent_manager, 'get_or_create_conversation'):
                            conversation = agent_manager.get_or_create_conversation(
                                source_agent.id, target_agent.id, create_new=True
                            )
                            conversation_id = conversation.conversation_id
                        else:
                            # Fallback to a simple ID for backward compatibility
                            conversation_id = f"{name1}_{name2}_conv"
                    
                    # Store for future reference - same ID for both directions
                    connection_conversation_ids[edge_pair] = conversation_id
                    connection_conversation_ids[reverse_edge_pair] = conversation_id
                
                # Create two edges for bidirectional display - one in each direction
                # First direction: name1 -> name2
                edges.append({
                    "data": {
                        "source": name1, 
                        "target": name2,
                        "is_new": is_new_connection,
                        "class": edge_class,
                        "conversation_id": conversation_id,
                        "distance": dist,
                        "direction": "forward"
                    }
                })
                
                # Second direction: name2 -> name1
                edges.append({
                    "data": {
                        "source": name2, 
                        "target": name1,
                        "is_new": is_new_connection,
                        "class": edge_class,
                        "conversation_id": conversation_id,
                        "distance": dist,
                        "direction": "backward"
                    }
                })
    
    return edges


def generate_elements(positions):
    """
    Generate Cytoscape elements (nodes and edges) from agent positions.
    """
    elements = []
    
    # Add nodes with movement probability classes and agent state info
    for name, pos in positions.items():
        # Calculate movement probability
        cooldown = agent_movement_cooldown.get(name, 0)
        probability = min(0.9, agent_movement_probability.get(name, 0.7) * (1 + 0.2 * cooldown))
        probability_percent = int(probability * 100)
        
        # Get the agent's current state
        agent = agent_lookup[name]
        agent_state = agent.state.name if hasattr(agent, 'state') else "IDLE"
        
        # Track if the agent is thinking for animation
        thinking = agent.thinking if hasattr(agent, 'thinking') else False
        
        # Assign a movement class based on probability
        movement_class = ""
        if cooldown == 0:
            movement_class = "just-joined"
        elif probability_percent > 70:
            movement_class = "likely-to-move"
        elif probability_percent > 40:
            movement_class = "may-move-soon"
        
        elements.append({
            "data": {
                "id": name, 
                "label": name,
                "movement_probability": probability,
                "class": movement_class,
                "state": agent_state,
                "thinking": thinking
            },
            "position": {"x": pos["x"], "y": pos["y"]}
        })
        
    # Add edges
    edges = compute_edges(positions)
    elements.extend(edges)
    return elements


def move_agent(agent_name, current_positions):
    """
    Move an agent to a new random position within grid bounds.
    """
    import random
    import logging
    
    logging.info(f"Moving agent {agent_name} from current position")
    
    # Current position
    current_x = current_positions[agent_name]["x"]
    current_y = current_positions[agent_name]["y"]
    
    # Random direction (0 = up, 1 = right, 2 = down, 3 = left, 4-7 = diagonals)
    direction = random.randint(0, 7)
    
    # Calculate new position based on direction
    if direction == 0:  # Up
        new_x, new_y = current_x, current_y - movement_distance
    elif direction == 1:  # Right
        new_x, new_y = current_x + movement_distance, current_y
    elif direction == 2:  # Down
        new_x, new_y = current_x, current_y + movement_distance
    elif direction == 3:  # Left
        new_x, new_y = current_x - movement_distance, current_y
    elif direction == 4:  # Up-Right
        new_x, new_y = current_x + movement_distance * 0.7, current_y - movement_distance * 0.7
    elif direction == 5:  # Down-Right
        new_x, new_y = current_x + movement_distance * 0.7, current_y + movement_distance * 0.7
    elif direction == 6:  # Down-Left
        new_x, new_y = current_x - movement_distance * 0.7, current_y + movement_distance * 0.7
    else:  # Up-Left
        new_x, new_y = current_x - movement_distance * 0.7, current_y - movement_distance * 0.7
    
    # Ensure the new position is within grid bounds
    new_x = max(grid_bounds["min_x"], min(grid_bounds["max_x"], new_x))
    new_y = max(grid_bounds["min_y"], min(grid_bounds["max_y"], new_y))
    
    # Make sure the agent actually moves by ensuring a minimum movement distance
    # If the new position is too close to the original, move it more
    distance = ((new_x - current_x)**2 + (new_y - current_y)**2)**0.5
    if distance < movement_distance * 0.5:
        # Force a larger movement
        new_x = current_x + random.choice([-1, 1]) * movement_distance
        new_y = current_y + random.choice([-1, 1]) * movement_distance
        
        # Ensure still within bounds
        new_x = max(grid_bounds["min_x"], min(grid_bounds["max_x"], new_x))
        new_y = max(grid_bounds["min_y"], min(grid_bounds["max_y"], new_y))
    
    logging.info(f"Agent {agent_name} moving from ({current_x}, {current_y}) to ({new_x}, {new_y})")
    
    return {"x": new_x, "y": new_y}


def simulation_step():
    """
    Process conversations between all connected agents.
    Each agent can now have multiple connections and conversations simultaneously.
    Update conversation logs accordingly and handle agent movement.
    """
    import random
    
    updates = []
    edges = compute_edges(agent_positions)
    
    # Track current connections to detect changes - use sets instead of dict for multiple connections
    current_connections = {}
    for edge in edges:
        source = edge["data"]["source"]
        target = edge["data"]["target"]
        
        # Add to sets of connections for each agent
        if source not in current_connections:
            current_connections[source] = set()
        if target not in current_connections:
            current_connections[target] = set()
            
        current_connections[source].add(target)
        current_connections[target].add(source)
    
    # Get previous connections from the agent's state
    previous_connections = getattr(simulation_step, 'previous_connections', {})
    
    # First, process conversations
    for edge in edges:
        source = edge["data"]["source"]
        target = edge["data"]["target"]
        
        # IMPORTANT: Ensure we never have agents talking to themselves
        if source == target:
            logging.warning(f"Skipping self-conversation: {source} -> {target}")
            continue
            
        # Get the agent objects
        source_agent = agent_lookup[source]
        target_agent = agent_lookup[target]
        
        # Check if this is a new connection
        previous_source_connections = previous_connections.get(source, set())
        previous_target_connections = previous_connections.get(target, set())
        
        is_new_connection = (target not in previous_source_connections) or (source not in previous_target_connections)
        
        # Get the conversation ID for this edge
        edge_pair = (source, target)
        if edge_pair in connection_conversation_ids:
            conversation_id = connection_conversation_ids[edge_pair]
        else:
            # Create a new conversation ID
            if hasattr(agent_manager, 'get_or_create_conversation'):
                conversation = agent_manager.get_or_create_conversation(
                    source_agent.id, target_agent.id, create_new=is_new_connection
                )
                conversation_id = conversation.conversation_id
            else:
                # Fallback for backward compatibility
                conversation_id = f"{source}_{target}_conv"
            
            # Store for future reference
            connection_conversation_ids[edge_pair] = conversation_id
        
        # Update conversation round counter
        conversation_pair = (source, target)
        if is_new_connection:
            conversation_rounds[conversation_pair] = 0
        else:
            conversation_rounds[conversation_pair] = conversation_rounds.get(conversation_pair, 0) + 1
        
        # IMPORTANT: Make sure we're not having agents talk to themselves
        if source == target:
            logging.warning(f"SKIPPING: Agent {source} is trying to talk to itself")
            continue
            
        if is_new_connection:
            # Reset agent movement cooldown for new connections
            agent_movement_cooldown[target] = 0
            
            # Notify agent of new connection
            notification = f"[SYSTEM: You are now connected to {source}. Please acknowledge with a greeting or continue the previous conversation.]"
            
            # Create a message object with the conversation ID
            notification_msg = Message(
                sender_id=0,  # 0 indicates system message
                content=notification,
                recipient_id=target_agent.id,
                conversation_id=conversation_id,
                is_system=True,
                requires_response=True  # New connection notifications DO require a response
            )
            
            # Add to the target agent's conversation
            target_agent.add_to_conversation(notification_msg, conversation_id)
            
            # Update agent state
            target_agent.set_state(Status.THINKING)
            
            # Get response from the agent using the integrated framework
            # Pass the conversation_id to maintain context
            notification_response = target_agent.generate_response(notification, conversation_id)
            
            # Check for movement in response
            if "[MOVE]" in notification_response:
                logging.info(f"{target_agent.name} wants to move from greeting")
                if hasattr(target_agent, 'framework_agent'):
                    # Add to action requests
                    if hasattr(target_agent.framework_agent, 'action_requests'):
                        target_agent.framework_agent.action_requests.append("move")
                    # Force move flag too
                    if hasattr(target_agent.framework_agent, '_force_move'):
                        target_agent.framework_agent._force_move = True
                # Remove the tag
                notification_response = notification_response.replace("[MOVE]", "").strip()
            
            # Update state and logs
            target_agent.set_state(Status.TALKING)
            conversation_logs[target].append(notification_response)
            updates.append((source, target, notification_response, conversation_id))
            
            # Create a response message object and add to the conversation
            response_msg = Message(
                sender_id=target_agent.id,
                content=notification_response,
                recipient_id=source_agent.id,
                conversation_id=conversation_id
            )
            target_agent.add_to_conversation(response_msg, conversation_id)
            
            # No need to add to both agents - this was causing duplicate messages
        else:
            # Normal communication flow - get the last message from the source to this specific target
            if conversation_id and hasattr(source_agent, 'get_conversation'):
                # Get the conversation between these agents
                conversation = source_agent.get_conversation(conversation_id)
                
                if conversation and conversation.messages:
                    # Find the most recent message the source agent sent to this target
                    for msg in reversed(conversation.messages):
                        if msg.sender_id == source_agent.id and msg.recipient_id == target_agent.id:
                            last_msg = msg.content
                            break
                    else:
                        # If we couldn't find a message, use a default greeting with target's name
                        last_msg = f"Hello {target}, I'd like to continue our conversation."
                else:
                    # No conversation exists yet, use a default greeting
                    last_msg = f"Hello {target}, nice to meet you!"
            # Fallback to using the framework agent's conversation history
            elif source_agent.framework_agent and source_agent.framework_agent.conversation_history:
                last_msg = source_agent.framework_agent.conversation_history[-1]["content"]
            else:
                last_msg = f"Hello {target}, how are you?"
            
            # Create a message object with the conversation ID
            message = Message(
                sender_id=source_agent.id,
                content=last_msg,
                recipient_id=target_agent.id,
                conversation_id=conversation_id
            )
            
            # Add to the target agent's conversation
            target_agent.add_to_conversation(message, conversation_id)
            
            # Update agent state
            target_agent.set_state(Status.THINKING)
            
            # Get response from the agent - pass conversation_id to maintain context
            response = target_agent.generate_response(last_msg, conversation_id)
            
            # Check for movement in response
            if "[MOVE]" in response:
                logging.info(f"{target_agent.name} wants to move")
                # Add a direct flag to the BaseAgent for more reliable movement detection
                target_agent._wants_to_move = True
                
                # Also add to framework agent's action requests for backward compatibility
                if hasattr(target_agent, 'framework_agent'):
                    # Add to action requests
                    if hasattr(target_agent.framework_agent, 'action_requests'):
                        # Make sure we don't duplicate the request
                        if "move" not in target_agent.framework_agent.action_requests:
                            target_agent.framework_agent.action_requests.append("move")
                    # Force move flag too
                    if hasattr(target_agent.framework_agent, '_force_move'):
                        target_agent.framework_agent._force_move = True
                
                # Remove the tag for display
                response = response.replace("[MOVE]", "").strip()
            
            # Update state and logs
            target_agent.set_state(Status.TALKING)
            conversation_logs[target].append(response)
            updates.append((source, target, response, conversation_id))
            
            # Create a response message object and add to the conversation
            response_msg = Message(
                sender_id=target_agent.id,
                content=response,
                recipient_id=source_agent.id,
                conversation_id=conversation_id
            )
            target_agent.add_to_conversation(response_msg, conversation_id)
            
            # No need to add to both agents - this was causing duplicate messages
    
    # Handle agent movement logic
    _handle_agent_movement(edges, previous_connections)
    
    # Store current connections for next step comparison
    simulation_step.previous_connections = current_connections
    return updates


async def simulation_step_async():
    """
    Asynchronous version of simulation_step that runs agent responses concurrently.
    Supports multiple conversations for each agent with all connected neighbors.
    """
    import random
    import asyncio
    import logging
    
    updates = []
    edges = compute_edges(agent_positions)
    
    # Track current connections to detect changes - use sets for multiple connections per agent
    current_connections = {}
    for edge in edges:
        source = edge["data"]["source"]
        target = edge["data"]["target"]
        
        # Add to sets of connections for each agent
        if source not in current_connections:
            current_connections[source] = set()
        if target not in current_connections:
            current_connections[target] = set()
            
        current_connections[source].add(target)
        current_connections[target].add(source)
    
    # Get previous connections from the agent's state
    previous_connections = getattr(simulation_step_async, 'previous_connections', {})
    
    # Process conversations asynchronously
    tasks = []
    for edge in edges:
        source = edge["data"]["source"]
        target = edge["data"]["target"]
        
        # IMPORTANT: Ensure we never have agents talking to themselves
        if source == target:
            logging.warning(f"Skipping self-conversation in async step: {source} -> {target}")
            continue
            
        # Get the agent objects
        source_agent = agent_lookup[source]
        target_agent = agent_lookup[target]
        
        # Check if this is a new connection
        previous_source_connections = previous_connections.get(source, set())
        previous_target_connections = previous_connections.get(target, set())
        
        is_new_connection = (target not in previous_source_connections) or (source not in previous_target_connections)
        
        # Get the conversation ID for this edge
        edge_pair = (source, target)
        if edge_pair in connection_conversation_ids:
            conversation_id = connection_conversation_ids[edge_pair]
        else:
            # Create a new conversation ID
            if hasattr(agent_manager, 'get_or_create_conversation'):
                conversation = agent_manager.get_or_create_conversation(
                    source_agent.id, target_agent.id, create_new=is_new_connection
                )
                conversation_id = conversation.conversation_id
            else:
                # Fallback for backward compatibility
                conversation_id = f"{source}_{target}_conv"
            
            # Store for future reference
            connection_conversation_ids[edge_pair] = conversation_id
        
        # Update conversation round counter
        conversation_pair = (source, target)
        if is_new_connection:
            conversation_rounds[conversation_pair] = 0
            agent_movement_cooldown[target] = 0  # Reset cooldown
            
            # Notify agent of new connection
            notification = f"[SYSTEM: You are now connected to {source}. Please acknowledge with a brief greeting.]"
            
            # Create a message object with the conversation ID
            notification_msg = Message(
                sender_id=0,  # 0 indicates system message
                content=notification,
                recipient_id=target_agent.id,
                conversation_id=conversation_id,
                is_system=True,
                requires_response=True  # New connection notifications DO require a response
            )
            
            # Add to the target agent's conversation
            target_agent.add_to_conversation(notification_msg, conversation_id)
            
            # Create task for asynchronous processing
            tasks.append((target_agent, notification, source, target, is_new_connection, conversation_id))
        else:
            conversation_rounds[conversation_pair] = conversation_rounds.get(conversation_pair, 0) + 1
            
            # Get the last message from the source to this specific target
            if conversation_id and hasattr(source_agent, 'get_conversation'):
                # Get the conversation between these agents
                conversation = source_agent.get_conversation(conversation_id)
                
                if conversation and conversation.messages:
                    # Find the most recent message the source agent sent to this target
                    for msg in reversed(conversation.messages):
                        if msg.sender_id == source_agent.id and msg.recipient_id == target_agent.id:
                            last_msg = msg.content
                            break
                    else:
                        # If we couldn't find a message, use a default greeting with target's name
                        last_msg = f"Hello {target}, I'd like to continue our conversation."
                else:
                    # No conversation exists yet, use a default greeting
                    last_msg = f"Hello {target}, nice to meet you!"
            # Fallback to using the framework agent's conversation history
            elif source_agent.framework_agent and source_agent.framework_agent.conversation_history:
                last_msg = source_agent.framework_agent.conversation_history[-1]["content"]
            else:
                last_msg = f"Hello {target}, how are you?"
            
            # Create a message object with the conversation ID
            message = Message(
                sender_id=source_agent.id,
                content=last_msg,
                recipient_id=target_agent.id,
                conversation_id=conversation_id
            )
            
            # Add to the target agent's conversation
            target_agent.add_to_conversation(message, conversation_id)
                
            # Create task for asynchronous processing
            tasks.append((target_agent, last_msg, source, target, is_new_connection, conversation_id))
    
    # Process all tasks concurrently
    if tasks:
        try:
            # The responses will be processed asynchronously and UI will be updated in real-time
            results = await asyncio.gather(*[_process_agent_response_async(agent, message, source, target, is_new, conv_id) 
                                for agent, message, source, target, is_new, conv_id in tasks])
            
            # Add the results to updates (filter out None responses)
            updates.extend([r for r in results if r[2] is not None])  # Check if response part is not None
        except Exception as e:
            logging.error(f"Error in async processing: {e}")
            # Continue even if there's an error
    
    # Handle agent movement
    _handle_agent_movement(edges, previous_connections)
    
    # Store current connections for next step comparison
    simulation_step_async.previous_connections = current_connections
    return updates


def _handle_agent_movement(edges, previous_connections):
    """
    Handle agent movement logic for both sync and async simulation step functions.
    Works with multiple connections per agent.
    """
    import random
    import logging  # Add logging
    
    # Determine which agents should move based on:
    # 1. Explicit movement requests
    # 2. Conversation rounds probabilistic movement
    agents_to_move = []
    
    # Debug log
    logging.info("Checking for agent movement requests...")
    
    # First check for explicit movement requests from all agents
    for name, agent in agent_lookup.items():
        # Get movement status from various sources
        wants_to_move = False
        
        # Check direct flag on the agent
        if hasattr(agent, '_wants_to_move') and agent._wants_to_move:
            wants_to_move = True
            logging.info(f"Agent {name} has _wants_to_move flag set to True")
            
        # Check action requests on framework agent
        if hasattr(agent, 'framework_agent') and hasattr(agent.framework_agent, 'action_requests'):
            if "move" in agent.framework_agent.action_requests:
                wants_to_move = True
                logging.info(f"Agent {name} has 'move' in action_requests: {agent.framework_agent.action_requests}")
            
        # Check force move flag on framework agent
        if hasattr(agent, 'framework_agent') and hasattr(agent.framework_agent, '_force_move'):
            if agent.framework_agent._force_move:
                wants_to_move = True
                logging.info(f"Agent {name} has _force_move flag set to True")
            
        # Also use the regular wants_to_move method as a fallback
        if not wants_to_move and agent.wants_to_move():
            wants_to_move = True
            logging.info(f"Agent {name} wants_to_move() returned True")
            
        # Process movement request if found
        if wants_to_move:
            logging.info(f"Agent {name} wants to move! Adding to movement list.")
            agents_to_move.append(name)
            
            # Clear all movement flags
            if hasattr(agent, '_wants_to_move'):
                agent._wants_to_move = False
                
            if hasattr(agent, 'framework_agent'):
                if hasattr(agent.framework_agent, 'action_requests') and "move" in agent.framework_agent.action_requests:
                    agent.framework_agent.action_requests.remove("move")
                    
                if hasattr(agent.framework_agent, '_force_move'):
                    agent.framework_agent._force_move = False
            
            # Log the explicit movement request
            movement_notification = f"[SYSTEM: {name} has explicitly requested to move to meet someone new.]"
            
            # Create a proper message object for this system notification
            movement_msg = Message(
                sender_id=0,  # 0 indicates system message
                content=movement_notification,
                recipient_id=agent.id,
                is_system=True,
                requires_response=False  # This notification doesn't need a response
                # No conversation ID needed as this is a standalone system message
            )
            
            # Add to the agent's active conversation if there is one
            if hasattr(agent, 'active_conversation_id') and agent.active_conversation_id:
                agent.add_to_conversation(movement_msg, agent.active_conversation_id)
            
            # Don't generate a response for movement notifications
            # They're just informational
            agent_movement_cooldown[name] = 0  # Reset cooldown
    
    # Get all conversation pairs from the edges
    all_conversation_pairs = set()
    for edge in edges:
        source = edge["data"]["source"]
        target = edge["data"]["target"]
        all_conversation_pairs.add((source, target))
    
    # Then check for probabilistic movement based on conversation duration across all connections
    for source, target in all_conversation_pairs:
        # Skip agents that already decided to move explicitly
        if source in agents_to_move or target in agents_to_move:
            continue
            
        conversation_pair = (source, target)
        rounds = conversation_rounds.get(conversation_pair, 0)
        
        # Increment movement cooldown for agents who have been talking for a while
        if rounds >= 2:  # After 2-3 rounds of conversation
            agent_movement_cooldown[source] += 1
            agent_movement_cooldown[target] += 1
    
    # Check if any agents should consider moving based on their overall conversation state
    for name, cooldown in agent_movement_cooldown.items():
        # Skip agents that already decided to move explicitly
        if name in agents_to_move:
            continue
            
        if cooldown >= 1:  # Agent has been in conversations for enough rounds
            # Probability increases the longer they've been talking
            probability = min(0.9, agent_movement_probability[name] * (1 + 0.2 * cooldown))
            
            # Roll for movement
            if random.random() < probability:
                agents_to_move.append(name)
                # Reset cooldown after deciding to move
                agent_movement_cooldown[name] = 0
    
    # Move agents who decided to move
    for agent_name in set(agents_to_move):  # Use set to avoid duplicates
        new_position = move_agent(agent_name, agent_positions)
        agent_positions[agent_name] = new_position
        
        # Log the movement
        movement_notification = f"[SYSTEM: {agent_name} has moved to a new location and will connect to new people in the next step.]"
        
        # Get the agent
        agent = agent_lookup[agent_name]
        
        # Create a proper message object for this system notification
        movement_msg = Message(
            sender_id=0,  # 0 indicates system message
            content=movement_notification,
            recipient_id=agent.id,
            is_system=True,
            requires_response=False  # This notification doesn't need a response
            # No conversation ID needed as this is a standalone system message
        )
        
        # Add to the agent's active conversation if there is one
        if hasattr(agent, 'active_conversation_id') and agent.active_conversation_id:
            agent.add_to_conversation(movement_msg, agent.active_conversation_id)
        
        # Don't generate a response for movement notifications
        # They're just informational


async def _process_agent_response_async(agent, message, source, target, is_new_connection, conversation_id):
    """
    Process a single agent response asynchronously
    """
    import logging
    # IMPORTANT: Make sure this target agent (who is responding) isn't the same as the source
    # This prevents agents from talking to themselves
    if agent.name == source:
        logging.warning(f"SKIPPING: Agent {agent.name} is trying to respond to itself")
        return source, target, None, conversation_id
    
    # Update agent state to thinking first (this will show the spinner)
    agent.set_state(Status.THINKING)
    agent.thinking = True
    
    # Generate response asynchronously - pass the conversation_id to maintain context
    response = await agent.generate_response_async(message, conversation_id)
    
    # Check for movement request in the response - improved version
    if "[MOVE]" in response:
        logging.info(f"{agent.name} wants to move (async)")
        # Add a direct flag to the BaseAgent for more reliable movement detection
        agent._wants_to_move = True
        
        # Also add to framework agent's action requests for backward compatibility
        if hasattr(agent, 'framework_agent'):
            if hasattr(agent.framework_agent, 'action_requests'):
                # Make sure we don't duplicate the request
                if "move" not in agent.framework_agent.action_requests:
                    agent.framework_agent.action_requests.append("move")
            # Force move flag too
            if hasattr(agent.framework_agent, '_force_move'):
                agent.framework_agent._force_move = True
                
        # Clean the response for display (remove the action tag)
        response = response.replace("[MOVE]", "").strip()
    
    # Update state and logs after getting response
    agent.set_state(Status.TALKING)
    agent.thinking = False
    # Only add non-None responses to the logs
    if response is not None:
        conversation_logs[agent.name].append(response)
    
    # Only add this message if it's valid
    if response is not None:
        # Create a response message object and add to the conversation
        source_agent = agent_lookup[source]
        response_msg = Message(
            sender_id=agent.id,
            content=response,
            recipient_id=source_agent.id,  # The recipient is the agent who sent the original message
            conversation_id=conversation_id
        )
        agent.add_to_conversation(response_msg, conversation_id)
    
    return source, target, response, conversation_id


# -------------------------
# Dash App Setup
# -------------------------
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
server = app.server

# Custom CSS for responsive design and white/orange theme
app.index_string = '''
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>SimuVerse - Agent Simulation</title>
        {%favicon%}
        {%css%}
        <style>
            :root {
                --primary-color: #FF7F00;
                --secondary-color: #FF9E33;
                --light-color: #FFF3E0;
                --text-color: #333333;
                --accent-color: #FF5722;
                --thinking-color: #9C27B0;
            }
            body {
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                background-color: white;
                color: var(--text-color);
                margin: 0;
                padding: 0;
            }
            .app-header {
                background: linear-gradient(135deg, var(--primary-color), var(--secondary-color));
                color: white;
                padding: 20px;
                text-align: center;
                box-shadow: 0 4px 6px rgba(0,0,0,0.1);
                margin-bottom: 20px;
            }
            .node-card {
                border-left: 4px solid var(--primary-color);
                background-color: var(--light-color);
                border-radius: 8px;
                padding: 15px;
                margin-bottom: 15px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.05);
                transition: transform 0.2s;
            }
            .node-card:hover {
                transform: translateY(-2px);
            }
            .node-card.thinking {
                border-left: 4px solid var(--thinking-color);
                animation: pulse 1.5s infinite;
            }
            @keyframes pulse {
                0% { box-shadow: 0 0 0 0 rgba(156, 39, 176, 0.4); }
                70% { box-shadow: 0 0 0 10px rgba(156, 39, 176, 0); }
                100% { box-shadow: 0 0 0 0 rgba(156, 39, 176, 0); }
            }
            .thinking-indicator {
                display: inline-block;
                width: 12px;
                height: 12px;
                border-radius: 50%;
                background-color: var(--thinking-color);
                margin-right: 5px;
                animation: blink 1s infinite;
            }
            @keyframes blink {
                0% { opacity: 0.4; }
                50% { opacity: 1; }
                100% { opacity: 0.4; }
            }
            .control-panel {
                background-color: white;
                border-radius: 8px;
                padding: 15px;
                box-shadow: 0 2px 8px rgba(0,0,0,0.08);
            }
            .simulation-btn {
                background-color: var(--primary-color);
                color: white;
                border: none;
                padding: 10px 20px;
                border-radius: 30px;
                font-weight: bold;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                transition: all 0.3s;
            }
            .simulation-btn:hover {
                background-color: var(--accent-color);
                transform: translateY(-2px);
                box-shadow: 0 4px 8px rgba(0,0,0,0.15);
            }
            .log-container {
                background-color: white;
                border-radius: 8px;
                padding: 15px;
                margin-top: 20px;
                box-shadow: 0 2px 8px rgba(0,0,0,0.08);
                max-height: 300px;
                overflow-y: auto;
            }
            /* Add spinner animation */
            .thinking-spinner {
                display: inline-block;
                width: 16px;
                height: 16px;
                border: 2px solid rgba(156, 39, 176, 0.3);
                border-radius: 50%;
                border-top-color: var(--thinking-color);
                animation: spinner 1s linear infinite;
                margin-left: 5px;
            }
            @keyframes spinner {
                to {transform: rotate(360deg);}
            }
            
            /* Chat style conversation history */
            .chat-container {
                padding: 15px;
                background-color: #f8f9fa;
                border-radius: 10px;
                box-shadow: 0 2px 8px rgba(0,0,0,0.05);
                height: 600px;
                overflow-y: auto;
                margin-top: 0;
            }
            .chat-title {
                font-size: 1.2rem;
                border-bottom: 1px solid #ddd;
                padding-bottom: 10px;
                margin-bottom: 15px;
                color: #333;
            }
            .chat-messages {
                display: flex;
                flex-direction: column;
            }
            .message {
                max-width: 75%;
                margin-bottom: 10px;
                padding: 12px 16px;
                border-radius: 18px;
                position: relative;
                line-height: 1.4;
                box-shadow: 0 1px 2px rgba(0,0,0,0.1);
                animation: fadeIn 0.3s ease-in-out;
            }
            
            @keyframes fadeIn {
                from { opacity: 0; transform: translateY(10px); }
                to { opacity: 1; transform: translateY(0); }
            }
            .message.left {
                align-self: flex-start;
                background-color: #f1f0f0;
                color: #333;
                border-bottom-left-radius: 5px;
            }
            .message.right {
                align-self: flex-end;
                background-color: var(--primary-color);
                color: white;
                border-bottom-right-radius: 5px;
                text-align: right;
                margin-left: auto;  /* This pushes the element to the right */
                margin-right: 0;
            }
            .message.system {
                align-self: center;
                background-color: #e1f5fe;
                color: #0277bd;
                border-radius: 10px;
                font-style: italic;
                max-width: 90%;
                text-align: center;
                font-size: 0.9rem;
            }
            .message-sender {
                font-weight: bold;
                margin-bottom: 3px;
                font-size: 0.85rem;
            }
            .message.left .message-sender {
                color: var(--accent-color);
            }
            .message.right .message-sender {
                color: #f8f9fa;
            }
            .chat-notification {
                font-size: 0.85rem;
                color: #666;
                text-align: center;
                margin: 10px 0;
                font-style: italic;
            }
            @media (max-width: 768px) {
                .container {
                    flex-direction: column;
                }
                .cytoscape-container {
                    height: 400px !important;
                }
                .message {
                    max-width: 90%;
                }
            }
        </style>
        <!-- Add CSS animations instead of JavaScript-based animations -->
        <style>
            /* CSS-only animation for thinking nodes */
            @keyframes thinking-pulse {
                0% { box-shadow: 0 0 5px 0px rgba(156, 39, 176, 0.3); }
                50% { box-shadow: 0 0 15px 5px rgba(156, 39, 176, 0.7); }
                100% { box-shadow: 0 0 5px 0px rgba(156, 39, 176, 0.3); }
            }
            
            @keyframes thinking-border-dash {
                to { stroke-dashoffset: 20; }
            }
            
            /* These will be applied via the stylesheet property in cytoscape */
            .thinking-node {
                animation: thinking-pulse 1.5s infinite;
            }
        </style>
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>
'''

app.layout = html.Div([
    # Hidden components for UI updates
    dcc.Interval(id='refresh-interval', interval=250, n_intervals=0),  # Refresh UI every 250ms to update thinking indicators
    
    # App Header with branding
    html.Div([
        html.H1("SimuVerse", className="mb-0"),
        html.P("Multi-Agent Interactive Simulation Environment", className="lead mb-0")
    ], className="app-header"),
    
    # Main content container
    dbc.Container([
        # Two columns for the entire layout
        dbc.Row([
            # Left column - Simulation visualization and chat
            dbc.Col([
                # Agent network visualization
                dbc.Card([
                    dbc.CardHeader(html.H4("Agent Network", className="text-center")),
                    dbc.CardBody([
                        cyto.Cytoscape(
                            id='cytoscape',
                            elements=generate_elements(agent_positions),
                            style={'width': '100%', 'height': '550px'},
                            layout={'name': 'preset'},
                            stylesheet=[
                                {'selector': 'node', 'style': {
                                    'label': 'data(label)',
                                    'background-color': '#FF7F00',
                                    'color': 'white',
                                    'text-outline-color': '#FF7F00',
                                    'text-outline-width': 2,
                                    'font-size': '14px',
                                    'text-valign': 'center',
                                    'text-halign': 'center',
                                    'width': 60, 
                                    'height': 60,
                                    'border-width': 3,
                                    'border-color': '#FF5722',
                                    'text-background-opacity': 1,
                                    'text-background-color': '#FF7F00',
                                    'text-background-shape': 'roundrectangle',
                                    'text-background-padding': '4px'
                                }},
                                # Thinking state styling with CSS animation
                                {'selector': 'node[state="THINKING"]', 'style': {
                                    'background-color': '#9C27B0',  # Purple for thinking state
                                    'border-width': 4,
                                    'border-color': '#E1BEE7', 
                                    'border-style': 'dashed',
                                    'border-opacity': 1,
                                    'border-dash-pattern': [6, 3],
                                    'text-background-color': '#9C27B0',
                                    'animation': 'thinking-pulse 1.5s infinite',
                                    'transition-property': 'background-color, border-color, border-width',
                                    'transition-duration': '0.3s'
                                }},
                                {'selector': 'edge', 'style': {
                                    'line-color': '#FFA500',
                                    'target-arrow-color': '#FF5722',
                                    'target-arrow-shape': 'triangle',
                                    'curve-style': 'bezier',
                                    'width': 3,
                                    'arrow-scale': 1.5,
                                    'opacity': 0.8,
                                    'z-index': 1  # Make sure edges appear below nodes
                                }},
                                {'selector': 'edge[?is_new]', 'style': {
                                    'line-color': '#FF3300',
                                    'target-arrow-color': '#FF3300',
                                    'width': 4,
                                    'line-style': 'dashed',
                                    'opacity': 1,
                                    'line-dash-pattern': [8, 3]
                                }},
                                {'selector': 'node[class="likely-to-move"]', 'style': {
                                    'border-width': 4,
                                    'border-color': '#FF3333',
                                    'border-style': 'dashed',
                                    'background-color': '#FF7F00',
                                    'border-opacity': 0.8
                                }},
                                {'selector': 'node[class="may-move-soon"]', 'style': {
                                    'border-width': 3,
                                    'border-color': '#FFC107',
                                    'border-style': 'dashed',
                                    'background-color': '#FF7F00',
                                    'border-opacity': 0.7
                                }},
                                {'selector': 'node[class="just-joined"]', 'style': {
                                    'border-width': 3,
                                    'border-color': '#33AAFF',
                                    'border-style': 'solid',
                                    'background-color': '#FF7F00',
                                    'border-opacity': 1
                                }},
                                {'selector': ':selected', 'style': {
                                    'background-color': '#FF5722',
                                    'line-color': '#FF5722',
                                    'border-width': 4,
                                    'border-color': '#FFC107',
                                    'opacity': 1
                                }}
                            ],
                            userPanningEnabled=True,
                            userZoomingEnabled=True,
                            autoungrabify=False,
                            minZoom=0.5,
                            maxZoom=2.0,
                            className="cytoscape-container"
                        )
                    ])
                ], className="mb-4 shadow-sm"),
                
                # Conversation history in chat style (now under the graph in left column)
                dbc.Card([
                    dbc.CardHeader([
                        html.H4([
                            html.I(className="fas fa-comments me-2"),
                            "Conversation History"
                        ], className="chat-title mb-0"),
                        html.Div(id="chat-title-details", className="text-muted small")
                    ]),
                    dbc.CardBody([
                        html.Div(id="chat-history", className="chat-messages")
                    ], className="chat-container")
                ], className="mt-4 shadow mb-4")
            ], md=8),
            
            # Right column - Controls and logs
            dbc.Col([
                # Control panel
                dbc.Card([
                    dbc.CardHeader(html.H4("Control Panel", className="text-center")),
                    dbc.CardBody([
                        dbc.Row([
                            dbc.Col(
                                dbc.Button("Step Simulation", id="step-btn", n_clicks=0, 
                                          className="simulation-btn w-100"),
                                width=6
                            ),
                            dbc.Col(
                                dbc.Button("Async Step", id="async-step-btn", n_clicks=0, 
                                          className="simulation-btn w-100", 
                                          color="secondary"),
                                width=6
                            )
                        ], className="mb-3"),
                        html.Div([
                            dbc.Alert([
                                html.I(className="fas fa-random me-2"),
                                "Autonomous Movement: Agents will move to new locations after talking to the same person for 2-3 rounds.",
                                html.Br(),
                                html.Small([
                                    html.Strong("Agent Tools: "),
                                    "Agents can decide to move by including ",
                                    html.Code("[MOVE]"),
                                    " in their responses."
                                ], className="mt-1 d-block")
                            ], color="info", className="py-2 mt-2 mb-3"),
                            
                            # Movement statistics
                            html.Div([
                                html.H6("Simulation Status", className="mb-2 border-bottom pb-1"),
                                html.Div(id="movement-stats", className="small")
                            ], className="mb-3"),
                            
                            html.P([
                                html.I(className="fas fa-info-circle me-2"),
                                "Drag nodes to manually reposition agents. Edges update based on proximity."
                            ], className="text-muted fst-italic small"),
                            html.P([
                                html.I(className="fas fa-exclamation-triangle me-2 text-warning"),
                                "When agents connect to new partners, they'll receive a notification and respond with a greeting.",
                                html.Br(),
                                html.Small("New connections are highlighted with dashed red lines.")
                            ], className="text-muted fst-italic small mt-2")
                        ])
                    ])
                ], className="mb-4 shadow-sm"),
                
                # Agent info panel
                dbc.CardBody(id="agent-info-panel", children=[
                    html.Div([
                        dbc.Row([
                            dbc.Col(html.Div([
                                # Agent name with conditional thinking indicator
                                html.Span(f"{name}", className="fw-bold"),
                                # Add a thinking spinner when agent is thinking
                                html.Span(id=f"thinking-indicator-{name}", className="thinking-spinner ms-2", 
                                         style={"display": "none"})
                            ]), width=3),
                            dbc.Col(html.Div([
                                html.Div(f"{agent.framework_agent.llm.__class__.__name__}" if agent.framework_agent else "No LLM", className="fw-bold"),
                                html.Div(f"{agent.framework_agent.llm if agent.framework_agent else ''}", className="text-muted small")       
                            ], className="d-flex flex-column"), width=3),
                            dbc.Col([
                                dbc.Row([
                                    html.Small("Memory", className="text-muted"),
                                    dcc.Slider(
                                        id={'type': 'memory-slider', 'index': name},
                                        min=0,
                                        max=1,
                                        step=1,
                                        value=1 if agent.framework_agent and agent.framework_agent.memory_enabled else 0,
                                        marks={0: 'Off', 1: 'On'},
                                        className="mb-2"
                                    )
                                ]),
                                dbc.Row([
                                    html.Small("Personality", className="text-muted"),
                                    dcc.Slider(
                                        id={'type': 'personality-slider', 'index': name},
                                        min=0,
                                        max=1,
                                        step=0.1,
                                        value=agent.framework_agent.personality_strength if agent.framework_agent else 0.5,
                                        marks={0: 'Low', 1: 'High'},
                                        className="mb-2"
                                    )
                                ])
                            ], width=6)
                        ], className="mb-2 node-card")
                        for name, agent in agent_lookup.items()
                    ])
                ]),
                
                # Status panel
                dbc.Card([
                    dbc.CardHeader(html.H4("Agent Status", className="text-center")),
                    dbc.CardBody([
                        html.P([
                            html.I(className="fas fa-info-circle me-2"),
                            "Click on a connection between agents to view their conversation below."
                        ], className="text-muted")
                    ])
                ], className="shadow-sm")
            ], md=4)
        ])
    ], fluid=True)
])


# We'll modify our approach to animation without using direct cy access
# Instead we'll rely on CSS animations for the thinking state

# -------------------------
# Callback: Update Graph When Nodes are Dragged or Step Simulation is Clicked
# -------------------------
@app.callback(
    [Output('cytoscape', 'elements'),
     *[Output(f"thinking-indicator-{name}", "style") for name in agent_lookup.keys()]],
    [Input('cytoscape', 'elements'),
     Input('step-btn', 'n_clicks'),
     Input('async-step-btn', 'n_clicks'),
     Input('cytoscape', 'tapNodeData')],
    [State('cytoscape', 'elements')]
)
def update_graph(current_elements, n_clicks, async_n_clicks, node_data, stored_elements):
    ctx = dash.callback_context
    triggered = ctx.triggered[0]['prop_id'] if ctx.triggered else ""

    # If nodes have been dragged, update agent_positions.
    # (Dash Cytoscape passes the updated element positions in the elements property.)
    if stored_elements:
        for ele in stored_elements:
            if 'position' in ele and 'id' in ele['data']:
                agent_positions[ele['data']['id']] = ele['position']

    # If step simulation button was clicked, run simulation step.
    if "step-btn" in triggered:
        # Run the synchronous simulation step
        simulation_step()
    
    # If async step button was clicked, run async simulation step
    if "async-step-btn" in triggered:
        # For Dash compatibility with async, we'll queue this function
        # to run in a separate thread/process, since Dash callbacks must be synchronous
        import threading
        import asyncio
        import nest_asyncio
        
        # Apply nest_asyncio to allow nested event loops (needed for Dash + asyncio)
        try:
            nest_asyncio.apply()
        except:
            # If nest_asyncio is not installed, print a warning but continue
            print("Warning: nest_asyncio not installed. Async button may not work properly.")
        
        # Create a thread that runs an event loop to execute the async function
        def run_async_step():
            try:
                # Try to get the running event loop
                loop = asyncio.get_event_loop()
            except RuntimeError:
                # If no event loop is running, create a new one
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
            
            # Run the async simulation step
            loop.run_until_complete(simulation_step_async())
            
            # Don't close the loop as it might be reused
        
        # Start the thread 
        thread = threading.Thread(target=run_async_step)
        thread.daemon = True
        thread.start()

    # Get current agent states for thinking indicators
    thinking_styles = []
    for name in agent_lookup.keys():
        agent = agent_lookup[name]
        # Show spinner if agent is thinking
        if hasattr(agent, 'thinking') and agent.thinking or agent.state == Status.THINKING:
            thinking_styles.append({"display": "inline-block"})
        else:
            thinking_styles.append({"display": "none"})

    # Return updated elements and thinking indicator styles
    return [generate_elements(agent_positions), *thinking_styles]


# We're replacing the old conversation log with a modern chat interface below

# -------------------------
# Callback: Update Conversation Title Details
# -------------------------
@app.callback(
    Output("chat-title-details", "children"),
    Input('cytoscape', 'tapEdgeData')
)
def update_chat_title(edgeData):
    if edgeData is None:
        return "Click on any connection to view a conversation"
        
    source = edgeData.get("source")
    target = edgeData.get("target")
    conversation_id = edgeData.get("conversation_id", "")
    
    # Check connection status
    current_connections = getattr(simulation_step, 'previous_connections', {})
    
    # Get the new connection status based on sets
    is_active_connection = False
    if isinstance(current_connections.get(source, None), set):
        is_active_connection = target in current_connections.get(source, set())
    elif isinstance(current_connections.get(target, None), set):
        is_active_connection = source in current_connections.get(target, set())
        
    connection_status = "Current Connection" if is_active_connection else "Previous Connection"
    
    # Get conversation details
    source_agent = agent_lookup[source]
    target_agent = agent_lookup[target]
    
    # Calculate message counts using conversation system if possible
    total_msgs = 0
    
    # Try to get message count from conversation if available
    if conversation_id and hasattr(source_agent, 'get_conversation'):
        conversation = source_agent.get_conversation(conversation_id)
        if conversation:
            total_msgs = len(conversation.get_messages())
        else:
            # Try target agent
            conversation = target_agent.get_conversation(conversation_id)
            if conversation:
                total_msgs = len(conversation.get_messages())
    
    # Fallback to legacy approach
    if total_msgs == 0:
        source_msgs = len(conversation_logs.get(source, []))
        target_msgs = len(conversation_logs.get(target, []))
        total_msgs = source_msgs + target_msgs
    
    # Format conversation ID for display
    conv_id_display = f"{conversation_id[:8]}..." if len(conversation_id) > 8 else conversation_id
    
    return [
        html.Span([
            html.I(className="fas fa-user-circle me-1"), 
            f"{source} ↔ {target}"  # Always show as bidirectional
        ], className="me-3"),
        html.Span([
            html.I(className="fas fa-exchange-alt me-1"),
            f"{connection_status}"
        ], className="me-3 badge bg-info text-white"),
        html.Span([
            html.I(className="fas fa-comment me-1"),
            f"{total_msgs} messages"
        ], className="badge bg-secondary text-white me-2"),
        html.Span([
            html.I(className="fas fa-fingerprint me-1"),
            f"ID: {conv_id_display}"
        ], className="badge bg-light text-dark")
    ]

# -------------------------
# Callback: Display Chat-style Conversation History
# -------------------------
@app.callback(
    Output("chat-history", "children"),
    [Input('cytoscape', 'tapEdgeData'),
     Input('refresh-interval', 'n_intervals')]
)
def display_chat_history(edgeData, n_intervals):
    if edgeData is None:
        return html.Div([
            html.Div("Click on a connection between agents to view their conversation.",
                    className="chat-notification")
        ])
    
    source = edgeData.get("source")
    target = edgeData.get("target")
    conversation_id = edgeData.get("conversation_id")
    
    # Create a unique chat container ID based on the conversation ID
    chat_container_id = f"chat-messages-container-{conversation_id}"
    
    # Get agents by name
    source_agent = agent_lookup.get(source)
    target_agent = agent_lookup.get(target)
    
    if not source_agent or not target_agent:
        return html.Div([
            html.Div(f"Could not find agents for {source} and {target}.",
                    className="chat-notification")
        ])
    
    messages = []
    
    # First try to get messages from the conversation ID
    if conversation_id:
        # Try to get conversation directly from one of the agents
        if hasattr(source_agent, 'get_conversation') and hasattr(target_agent, 'get_conversation'):
            source_conversation = source_agent.get_conversation(conversation_id)
            
            if source_conversation:
                # Get all messages from this conversation
                messages = source_conversation.get_messages()
            else:
                target_conversation = target_agent.get_conversation(conversation_id)
                if target_conversation:
                    messages = target_conversation.get_messages()
    
    # If no messages found through conversation ID, fall back to the old method
    if not messages:
        # Fallback: use the conversation logs
        source_msgs = conversation_logs.get(source, [])
        target_msgs = conversation_logs.get(target, [])
        
        # Legacy approach for backward compatibility
        chat_messages = []
        chat_messages.append(
            html.Div(
                f"{source} and {target} are connected",
                className="chat-notification"
            )
        )
        
        # Get the maximum number of messages between the two
        max_msgs = max(len(source_msgs), len(target_msgs))
        
        # Interleave messages - each round has a message from source followed by target
        for i in range(max_msgs):
            # Add source message if available
            if i < len(source_msgs):
                msg = source_msgs[i]
                is_system = '[SYSTEM:' in msg
                
                if is_system:
                    # System message
                    chat_messages.append(
                        html.Div(
                            msg.replace('[SYSTEM:', '').replace(']', ''),
                            className="message system"
                        )
                    )
                else:
                    # Regular message from source - positioned on left
                    chat_messages.append(
                        html.Div([
                            html.Div(source, className="message-sender"),
                            html.Div(msg)
                        ], className="message left")
                    )
            
            # Add target message if available
            if i < len(target_msgs):
                msg = target_msgs[i]
                is_system = '[SYSTEM:' in msg
                
                if is_system:
                    # System message
                    chat_messages.append(
                        html.Div(
                            msg.replace('[SYSTEM:', '').replace(']', ''),
                            className="message system"
                        )
                    )
                else:
                    # Regular message from target
                    chat_messages.append(
                        html.Div([
                            html.Div(target, className="message-sender"),
                            html.Div(msg)
                        ], className="message right")
                    )
    else:
        # Use new conversation system
        chat_messages = []
        
        # Add a connection notification with helpful debugging info
        chat_messages.append(
            html.Div([
                html.Div(f"{source} and {target} are connected", className="mb-1"),
                html.Div(f"Conversation ID: {conversation_id[:8]}...", className="small text-muted")
            ], className="chat-notification")
        )
        
        # Sort messages by order they were added
        sorted_messages = sorted(messages, key=lambda m: messages.index(m))
        
        # Display all messages in the conversation
        for msg in sorted_messages:
            content = msg.content
            sender_id = msg.sender_id
            
            # Skip empty messages or None values (could happen with async responses)
            if not content or content is None:
                continue
                
            # Determine the sender name
            if sender_id == 0:
                # System message
                is_system = True
                sender_name = "SYSTEM"
            else:
                is_system = False
                # Find the agent name from the ID
                sender_name = next((name for name, agent in agent_lookup.items() 
                                  if agent.id == sender_id), f"Agent_{sender_id}")
            
            if is_system or sender_name == "SYSTEM" or '[SYSTEM:' in content:
                # System message
                system_content = content
                if '[SYSTEM:' in content:
                    system_content = content.replace('[SYSTEM:', '').replace(']', '')
                    
                chat_messages.append(
                    html.Div(
                        system_content,
                        className="message system"
                    )
                )
            else:
                # For clarity, always show as left/right based on which side the agent is on in the UI
                # This ensures consistent positioning for the whole conversation
                if sender_name == source:
                    position = "left"
                else:
                    position = "right"
                    
                chat_messages.append(
                    html.Div([
                        html.Div(sender_name, className="message-sender"),
                        html.Div(content)
                    ], className=f"message {position}")
                )
    
    # Add JavaScript to auto-scroll to the bottom of conversation
    container_with_scroll = html.Div(
        chat_messages,
        id=chat_container_id,
        # Auto-scroll to bottom with JavaScript
        style={
            "height": "100%",
            "overflow-y": "auto"
        }
    )
    
    # Add a script to scroll to bottom
    return [
        container_with_scroll,
        html.Script(f"""
            // Wait a short time for rendering to complete
            setTimeout(function() {{
                var chatContainer = document.getElementById('{chat_container_id}');
                if (chatContainer) {{
                    // Scroll to the bottom to show the latest messages
                    chatContainer.scrollTop = chatContainer.scrollHeight;
                    
                    // Add a MutationObserver to scroll down when new messages are added
                    var observer = new MutationObserver(function(mutations) {{
                        chatContainer.scrollTop = chatContainer.scrollHeight;
                    }});
                    
                    // Start observing the chat container for DOM changes
                    observer.observe(chatContainer, {{ childList: true, subtree: true }});
                }}
            }}, 100);
        """)
    ]


# Removed old log style updater - no longer needed

# Update the thinking indicators based on the interval
@app.callback(
    [*[Output(f"thinking-indicator-{name}", "style", allow_duplicate=True) for name in agent_lookup.keys()]],
    Input("refresh-interval", "n_intervals"),
    prevent_initial_call=True
)
def update_thinking_indicators(n_intervals):
    """Update the thinking indicators based on agent state"""
    thinking_styles = []
    for name in agent_lookup.keys():
        agent = agent_lookup[name]
        # Show spinner if agent is thinking
        if hasattr(agent, 'thinking') and agent.thinking or agent.state == Status.THINKING:
            thinking_styles.append({"display": "inline-block"})
        else:
            thinking_styles.append({"display": "none"})
    return thinking_styles

# Movement statistics update
@app.callback(
    Output("movement-stats", "children"),
    Input("step-btn", "n_clicks"),
    Input("cytoscape", "elements")
)
def update_movement_stats(n_clicks, elements):
    """Update the movement statistics display"""
    # Get the current edges
    edges = [ele for ele in elements if "source" in ele.get("data", {})]
    
    # Process edges to avoid duplicate connections in display
    # Use the conversation_id to identify unique connections
    processed_connections = set()
    
    # Format connection information
    connections = []
    for edge in edges:
        source = edge["data"]["source"]
        target = edge["data"]["target"]
        conversation_id = edge["data"].get("conversation_id", "")
        
        # Skip if we've already processed this conversation
        if conversation_id in processed_connections:
            continue
            
        # Add to processed set
        processed_connections.add(conversation_id)
        
        # Get conversation rounds for display
        rounds = conversation_rounds.get((source, target), 0)
        
        # Add bidirectional connection indicator (↔)
        connections.append(html.Div([
            html.Span([
                html.Span(f"{source}", className="fw-bold text-warning"), 
                " ↔ ",  # Show as a two-way connection
                html.Span(f"{target}", className="fw-bold text-warning")
            ]),
            html.Span(f" ({rounds} rounds)", className="ms-2 text-muted")
        ], className="mb-1"))
    
    # Format movement status
    movement_info = []
    
    # Check for movement requests - now using the agent manager
    # Store movement requests in a persistent dictionary to prevent them from disappearing
    if not hasattr(update_movement_stats, 'pending_moves'):
        update_movement_stats.pending_moves = {}
        
    # First update any existing pending moves
    for name in list(update_movement_stats.pending_moves.keys()):
        # Remove items that have timed out (5 updates without movement)
        if update_movement_stats.pending_moves[name] > 5:
            del update_movement_stats.pending_moves[name]
        else:
            # Increment the counter for existing pending moves
            update_movement_stats.pending_moves[name] += 1
            
    # Check for new movement requests
    movement_requests = []
    for name, agent in agent_lookup.items():
        if agent.wants_to_move():
            # Reset the counter for this agent's move request
            update_movement_stats.pending_moves[name] = 0
            movement_requests.append(name)
    
    # Combine active movement requests with pending ones
    all_movement_requests = list(set(movement_requests + list(update_movement_stats.pending_moves.keys())))
    
    for name, cooldown in agent_movement_cooldown.items():
        probability = min(0.9, agent_movement_probability[name] * (1 + 0.2 * cooldown))
        probability_percent = int(probability * 100)
        
        # Determine status text and color
        if name in all_movement_requests:
            status = "Requesting to move"
            color = "text-primary fw-bold"
            icon = html.I(className="fas fa-walking me-1")
        elif cooldown == 0:
            status = "Just joined"
            color = "text-info"
            icon = ""
        elif probability_percent > 70:
            status = "Likely to move"
            color = "text-danger"
            icon = ""
        elif probability_percent > 40:
            status = "May move soon"
            color = "text-warning"
            icon = ""
        else:
            status = "Staying"
            color = "text-success"
            icon = ""
            
        movement_info.append(html.Div([
            html.Span(f"{name}: ", className="fw-bold"),
            icon,
            html.Span(f"{status} ", className=f"{color}"),
            html.Span(f"({probability_percent}%)", className="text-muted small")
        ], className="mb-1"))
    
    return html.Div([
        html.Div([
            html.H6("Conversation Rounds", className="mb-1 small text-secondary"),
            html.Div(connections)
        ], className="mb-3"),
        html.Div([
            html.H6("Movement Status", className="mb-1 small text-secondary"),
            html.Div(movement_info)
        ]),
    ])

@app.callback(
    Output('cytoscape', 'elements', allow_duplicate=True),
    [Input({'type': 'memory-slider', 'index': ALL}, 'value'),
     Input({'type': 'personality-slider', 'index': ALL}, 'value')],
    [State({'type': 'memory-slider', 'index': ALL}, 'id'),
     State({'type': 'personality-slider', 'index': ALL}, 'id'),
     State('cytoscape', 'elements')],
    prevent_initial_call=True
)
def update_agent_settings(memory_values, personality_values, memory_ids, personality_ids, elements):
    ctx = dash.callback_context
    if not ctx.triggered:
        return elements
        
    # Update memory settings
    for slider_id, value in zip(memory_ids, memory_values):
        agent_name = slider_id['index']
        agent_lookup[agent_name].set_memory_enabled(bool(value))
        
    # Update personality settings
    for slider_id, value in zip(personality_ids, personality_values):
        agent_name = slider_id['index']
        agent_lookup[agent_name].set_personality_strength(float(value))
        
    return elements  # Return existing elements to refresh display

if __name__ == '__main__':
    app.run(debug=True, port=8050)
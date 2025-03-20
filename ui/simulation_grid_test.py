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
from src.agent_manager import create_agent_with_llm, Agent, AgentManager, Status

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
james = create_agent_with_llm(
    agent_id=1,
    name="James",
    provider="openai",
    api_key=openai_api_key,
    model="gpt-4o-mini",
    system_prompt=(
        "You are James, a friendly 20 yr old male college student in a social simulation. "
        "Respond naturally in a conversational tone and limit your reply to no more than 2 sentences. "
        "After talking to the same person for 2-3 rounds, you prefer to move and meet someone new. "
        "You're curious and enjoy meeting different people. "
        "When you want to move to meet someone new, include the exact text [MOVE] somewhere in your response. "
        "This will cause you to physically move in the simulation to meet someone else."
    ),
    memory_enabled=True,
    personality_strength=0.7
)

jade = create_agent_with_llm(
    agent_id=2,
    name="Jade",
    provider="claude",
    api_key=claude_api_key,
    model="claude-3-5-haiku-20241022",
    system_prompt=(
        "You are Jade, an engaging 20 yr old female computer scientist in a social simulation. "
        "Respond concisely and in a human-like manner in no more than 2 sentences. "
        "After talking to the same person for 2-3 exchanges, you prefer to move around and meet new people. "
        "You're outgoing and enjoy diverse conversations. "
        "When you want to move to meet someone new, include the exact text [MOVE] somewhere in your response. "
        "This will cause you to physically move in the simulation to meet someone else."
    ),
    memory_enabled=True,
    personality_strength=0.5
)

jesse = create_agent_with_llm(
    agent_id=3,
    name="Jesse",
    provider="claude",
    api_key=claude_api_key,
    model="claude-3-5-haiku-20241022",
    system_prompt=(
        "You are Jesse, a 20 yr old male soldier from South Korea in a social simulation. "
        "Respond concisely and in a human-like manner in no more than 2 sentences. "
        "After talking to the same person for 2-3 exchanges, you like to move to a new location and meet different people. "
        "You're disciplined but enjoy socializing with various individuals. "
        "When you want to move to meet someone new, include the exact text [MOVE] somewhere in your response. "
        "This will cause you to physically move in the simulation to meet someone else."
    ),
    memory_enabled=True,
    personality_strength=0.9
)

jamal = create_agent_with_llm(
    agent_id=4,
    name="Jamal",
    provider="openai",
    api_key=openai_api_key,
    model="gpt-4o-mini",
    system_prompt=(
        "You are Jamal, a 20 yr old male electrician working at NASA in a social simulation. "
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
agents = [james, jade, jesse, jamal]
agent_manager = AgentManager(agents)

# Dictionary mapping agent names to agent objects for lookup
agent_lookup = {
    "James": james,
    "Jade": jade,
    "Jesse": jesse,
    "Jamal": jamal
}

# -------------------------
# Simulation Data Structures
# -------------------------
# Initial agent positions (could be randomized or pre-set)
agent_positions = {
    "James": {"x": 100, "y": 200},
    "Jade": {"x": 300, "y": 200},
    "Jesse": {"x": 100, "y": 400},
    "Jamal": {"x": 300, "y": 400}
}

# For simplicity, maintain a dictionary for conversation logs
conversation_logs = {name: [] for name in agent_lookup.keys()}

# Track conversation rounds between agents
conversation_rounds = {}  # Format: {(source, target): count}

# Track movement state
agent_movement_cooldown = {name: 0 for name in agent_lookup.keys()}  # Countdown until agent considers moving
agent_movement_probability = {name: 0.7 for name in agent_lookup.keys()}  # Base probability of movement
grid_bounds = {"min_x": 50, "max_x": 550, "min_y": 50, "max_y": 550}  # Grid boundaries
movement_distance = 75  # How far agents move in one step


def compute_edges(positions):
    """
    Compute edges based on proximity.
    For each agent, connect it to its nearest neighbor.
    """
    edges = []
    names = list(positions.keys())
    
    # Get previous connections to detect changes
    previous_connections = getattr(simulation_step, 'previous_connections', {}) if 'simulation_step' in globals() else {}
    
    for name in names:
        best = None
        best_dist = float("inf")
        for other in names:
            if other == name:
                continue
            dx = positions[name]["x"] - positions[other]["x"]
            dy = positions[name]["y"] - positions[other]["y"]
            dist = math.sqrt(dx * dx + dy * dy)
            if dist < best_dist:
                best_dist = dist
                best = other
        if best:
            # Check if this is a new connection
            is_new_connection = previous_connections.get(best) != name
            
            # Add visual class to mark new connections
            edge_class = "new-connection" if is_new_connection else ""
            
            edges.append({
                "data": {
                    "source": name, 
                    "target": best,
                    "is_new": is_new_connection,
                    "class": edge_class
                }
            })
    return edges


def generate_elements(positions):
    """
    Generate Cytoscape elements (nodes and edges) from agent positions.
    """
    elements = []
    
    # Add nodes with movement probability classes
    for name, pos in positions.items():
        # Calculate movement probability
        cooldown = agent_movement_cooldown.get(name, 0)
        probability = min(0.9, agent_movement_probability.get(name, 0.7) * (1 + 0.2 * cooldown))
        probability_percent = int(probability * 100)
        
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
                "class": movement_class
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
    
    return {"x": new_x, "y": new_y}


def simulation_step():
    """
    For each computed edge (source -> target), take the last message from the source
    and pass it to the target agent. Update conversation logs accordingly.
    Also handles agent movement after sufficient conversation rounds.
    """
    import random
    
    updates = []
    edges = compute_edges(agent_positions)
    
    # Track current connections to detect changes
    current_connections = {}
    for edge in edges:
        source = edge["data"]["source"]
        target = edge["data"]["target"]
        current_connections[target] = source
    
    # Get previous connections from the agent's state
    previous_connections = getattr(simulation_step, 'previous_connections', {})
    
    # First, process conversations
    for edge in edges:
        source = edge["data"]["source"]
        target = edge["data"]["target"]
        
        # Get the agent objects
        source_agent = agent_lookup[source]
        target_agent = agent_lookup[target]
        
        # Check if this is a new connection
        is_new_connection = previous_connections.get(target) != source
        
        # Update conversation round counter
        conversation_pair = (source, target)
        if is_new_connection:
            conversation_rounds[conversation_pair] = 0
        else:
            conversation_rounds[conversation_pair] = conversation_rounds.get(conversation_pair, 0) + 1
        
        if is_new_connection:
            # Reset agent movement cooldown for new connections
            agent_movement_cooldown[target] = 0
            
            # Notify agent of new connection
            notification = f"[SYSTEM: You are now connected to {source}. Please acknowledge with a brief greeting.]"
            
            # Update agent state
            target_agent.set_state(Status.THINKING)
            
            # Get response from the agent using the integrated framework
            notification_response = target_agent.generate_response(notification)
            
            # Update state and logs
            target_agent.set_state(Status.TALKING)
            conversation_logs[target].append(notification_response)
            updates.append((source, target, notification_response))
        else:
            # Normal communication flow - get the last message from the source agent
            # We should use the framework agent's history, but for simplicity we'll pass the last response
            if source_agent.framework_agent and source_agent.framework_agent.conversation_history:
                last_msg = source_agent.framework_agent.conversation_history[-1]["content"]
            else:
                last_msg = "Hello"
            
            # Update agent state
            target_agent.set_state(Status.THINKING)
            
            # Get response from the agent
            response = target_agent.generate_response(last_msg)
            
            # Update state and logs
            target_agent.set_state(Status.TALKING)
            conversation_logs[target].append(response)
            updates.append((source, target, response))
    
    # Second, determine which agents should move based on:
    # 1. Explicit movement requests
    # 2. Conversation rounds probabilistic movement
    agents_to_move = []
    
    # First check for explicit movement requests from all agents
    for name, agent in agent_lookup.items():
        if agent.wants_to_move():
            agents_to_move.append(name)
            # Log the explicit movement request
            movement_notification = f"[SYSTEM: {name} has explicitly requested to move to meet someone new.]"
            agent.generate_response(movement_notification)
            agent_movement_cooldown[name] = 0  # Reset cooldown
    
    # Then check for probabilistic movement based on conversation duration
    for edge in edges:
        source = edge["data"]["source"]
        target = edge["data"]["target"]
        
        # Skip agents that already decided to move explicitly
        if source in agents_to_move or target in agents_to_move:
            continue
            
        conversation_pair = (source, target)
        rounds = conversation_rounds.get(conversation_pair, 0)
        
        # Increment movement cooldown for agents who have been talking for a while
        if rounds >= 2:  # After 2-3 rounds of conversation
            agent_movement_cooldown[source] += 1
            agent_movement_cooldown[target] += 1
        
        # Check if agents should consider moving
        for agent_name in [source, target]:
            if agent_name in agents_to_move:
                continue  # Skip if already moving
                
            if agent_movement_cooldown[agent_name] >= 1:  # Agent has been in a conversation for enough rounds
                # Probability increases the longer they've been talking to the same person
                probability = min(0.9, agent_movement_probability[agent_name] * (1 + 0.2 * agent_movement_cooldown[agent_name]))
                
                # Roll for movement
                if random.random() < probability:
                    agents_to_move.append(agent_name)
                    # Reset cooldown after deciding to move
                    agent_movement_cooldown[agent_name] = 0
    
    # Third, move agents who decided to move
    for agent_name in set(agents_to_move):  # Use set to avoid duplicates
        new_position = move_agent(agent_name, agent_positions)
        agent_positions[agent_name] = new_position
        
        # Log the movement
        movement_notification = f"[SYSTEM: {agent_name} has moved to a new location and will connect to a new person in the next step.]"
        agent_lookup[agent_name].generate_response(movement_notification)
    
    # Store current connections for next step comparison
    simulation_step.previous_connections = current_connections
    return updates


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
            @media (max-width: 768px) {
                .container {
                    flex-direction: column;
                }
                .cytoscape-container {
                    height: 400px !important;
                }
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
    # App Header with branding
    html.Div([
        html.H1("SimuVerse", className="mb-0"),
        html.P("Multi-Agent Interactive Simulation Environment", className="lead mb-0")
    ], className="app-header"),
    
    # Main content container
    dbc.Container([
        dbc.Row([
            # Left column - Simulation visualization
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader(html.H4("Agent Network", className="text-center")),
                    dbc.CardBody([
                        cyto.Cytoscape(
                            id='cytoscape',
                            elements=generate_elements(agent_positions),
                            style={'width': '100%', 'height': '600px'},
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
                                {'selector': 'edge', 'style': {
                                    'line-color': '#FFA500',
                                    'target-arrow-color': '#FF5722',
                                    'target-arrow-shape': 'triangle',
                                    'curve-style': 'bezier',
                                    'width': 3,
                                    'arrow-scale': 1.5,
                                    'opacity': 0.8
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
            ], md=8),
            
            # Right column - Controls and logs
            dbc.Col([
                # Control panel
                dbc.Card([
                    dbc.CardHeader(html.H4("Control Panel", className="text-center")),
                    dbc.CardBody([
                        dbc.Button("Step Simulation", id="step-btn", n_clicks=0, 
                                  className="simulation-btn w-100 mb-3"),
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
                dbc.CardBody([
                    html.Div([
                        dbc.Row([
                            dbc.Col(html.Div(f"{name}", className="fw-bold"), width=3),
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
                
                # Conversation log panel
                dbc.Card([
                    dbc.CardHeader(html.H4("Conversation Logs", className="text-center")),
                    dbc.CardBody([
                        html.Div(id="log-div", className="log-container")
                    ])
                ], className="shadow-sm")
            ], md=4)
        ])
    ], fluid=True)
])


# -------------------------
# Callback: Update Graph When Nodes are Dragged or Step Simulation is Clicked
# -------------------------
@app.callback(
    Output('cytoscape', 'elements'),
    Input('cytoscape', 'elements'),
    Input('step-btn', 'n_clicks'),
    State('cytoscape', 'elements')
)
def update_graph(current_elements, n_clicks, stored_elements):
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
        # Run the simulation step - this now includes the autonomous movement logic
        simulation_step()

    # Return updated elements (with recomputed edges)
    return generate_elements(agent_positions)


# -------------------------
# Callback: Display Logs When an Edge is Clicked with Improved Formatting
# -------------------------
@app.callback(
    Output("log-div", "children"),
    Input('cytoscape', 'tapEdgeData')
)
def display_logs(edgeData):
    if edgeData is None:
        return html.Div([
            html.P("Click on an edge to view conversation logs.", 
                  className="text-center text-muted fst-italic"),
            html.Div(className="text-center", children=[
                html.I(className="fas fa-arrow-left fa-2x text-warning")
            ])
        ])
    
    source = edgeData.get("source")
    target = edgeData.get("target")
    
    # Check connection status
    current_connections = getattr(simulation_step, 'previous_connections', {})
    connection_status = "Current Connection" if current_connections.get(target) == source else "Previous Connection"
    
    return html.Div([
        html.H5([
            "Conversation: ", 
            html.Span(f"{source}", className="text-warning fw-bold"), 
            " → ", 
            html.Span(f"{target}", className="text-warning fw-bold"),
            html.Span(f" ({connection_status})", className="ms-2 badge bg-info text-white small")
        ], className="mb-3"),
        
        # Source agent logs
        html.Div([
            html.H6([
                html.Span(f"{source}", className="badge bg-warning text-dark me-2"),
                "Messages"
            ], className="border-bottom pb-2"),
            html.Div([
                html.Div(msg, className="p-2 mb-2 bg-light rounded") 
                for msg in conversation_logs.get(source, ["No messages yet"])
            ])
        ], className="mb-3"),
        
        # Target agent logs
        html.Div([
            html.H6([
                html.Span(f"{target}", className="badge bg-warning text-dark me-2"),
                "Responses"
            ], className="border-bottom pb-2"),
            html.Div([
                # Style notifications differently
                html.Div(
                    msg, 
                    className=f"p-2 mb-2 {'bg-warning bg-opacity-25' if '[SYSTEM:' in msg else 'bg-light'} rounded"
                ) for msg in conversation_logs.get(target, ["No messages yet"])
            ])
        ])
    ])


# Add real-time statistics update
@app.callback(
    Output("log-div", "style"),
    Input("log-div", "children")
)
def update_log_style(children):
    """Make sure log container maintains proper styling with dynamic content"""
    return {
        "background-color": "white",
        "border-radius": "8px",
        "padding": "15px",
        "max-height": "300px",
        "overflow-y": "auto",
        "box-shadow": "0 2px 8px rgba(0,0,0,0.08)",
        "transition": "all 0.3s"
    }

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
    
    # Format connection information
    connections = []
    for edge in edges:
        source = edge["data"]["source"]
        target = edge["data"]["target"]
        rounds = conversation_rounds.get((source, target), 0)
        connections.append(html.Div([
            html.Span([
                html.Span(f"{source}", className="fw-bold text-warning"), 
                " → ", 
                html.Span(f"{target}", className="fw-bold text-warning")
            ]),
            html.Span(f" ({rounds} rounds)", className="ms-2 text-muted")
        ], className="mb-1"))
    
    # Format movement status
    movement_info = []
    
    # Check for movement requests - now using the agent manager
    movement_requests = []
    for name, agent in agent_lookup.items():
        if agent.wants_to_move():
            movement_requests.append(name)
    
    for name, cooldown in agent_movement_cooldown.items():
        probability = min(0.9, agent_movement_probability[name] * (1 + 0.2 * cooldown))
        probability_percent = int(probability * 100)
        
        # Determine status text and color
        if name in movement_requests:
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
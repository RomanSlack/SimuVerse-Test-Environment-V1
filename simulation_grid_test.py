import streamlit as st
import os
from dotenv import load_dotenv
import math
import dash
from dash import html, dcc, Input, Output, State
import dash_cytoscape as cyto
from dash import dash_table
import dash_bootstrap_components as dbc
from modules.framework import create_agent
# -------------------------
# Load API Keys and Create Agents
# -------------------------
load_dotenv(override=True)
openai_api_key = os.getenv("OPENAI_API_KEY")


claude_api_key = os.getenv("CLAUDE_API_KEY")
if not openai_api_key or not claude_api_key:
    st.error("One or more API keys not found in environment variables.")
    st.stop()

# Create several agents
james = create_agent(
    provider="openai",
    name="James",
    api_key=openai_api_key,
    model="gpt-4o-mini",
    system_prompt=(
        "You are James, a friendly 20 yr old male college student. "
        "Respond naturally in a conversational tone and limit your reply to no more than 2 sentences."
    )
)

jade = create_agent(
    provider="claude",
    name="Jade",
    api_key=claude_api_key,
    model="claude-3-5-haiku-20241022",
    system_prompt=(
        "You are Jade, an engaging 20 yr old female computer scientist. "
        "Respond concisely and in a human-like manner in no more than 2 sentences."
    )
)

jesse = create_agent(
    provider="claude",
    name="Jesse",
    api_key=claude_api_key,
    model="claude-3-5-haiku-20241022",
    system_prompt=(
        "You are Jesse, a 20 yr old male soldier from South Korea. "
        "Respond concisely and in a human-like manner in no more than 2 sentences."
    )
)

jamal = create_agent(
    provider="openai",
    name="Jamal",
    api_key=openai_api_key,
    model="gpt-4o-mini",
    system_prompt=(
        "You are Jamal, a 20 yr old male electrician working at NASA. "
        "Respond naturally in a conversational tone and limit your reply to no more than 2 sentences."
    )
)


# Dictionary mapping agent names to agent objects
agents = {
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

# For simplicity, maintain a dictionary for conversation logs.
conversation_logs = {name: [] for name in agents.keys()}


def compute_edges(positions):
    """
    Compute edges based on proximity.
    For each agent, connect it to its nearest neighbor.
    """
    edges = []
    names = list(positions.keys())
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
            edges.append({"data": {"source": name, "target": best}})
    return edges


def generate_elements(positions):
    """
    Generate Cytoscape elements (nodes and edges) from agent positions.
    """
    elements = []
    for name, pos in positions.items():
        elements.append({
            "data": {"id": name, "label": name},
            "position": {"x": pos["x"], "y": pos["y"]}
        })
    edges = compute_edges(positions)
    elements.extend(edges)
    return elements


def simulation_step():
    """
    For each computed edge (source -> target), take the last message from the source
    and pass it to the target agent. Update conversation logs accordingly.
    """
    updates = []
    edges = compute_edges(agent_positions)
    for edge in edges:
        source = edge["data"]["source"]
        target = edge["data"]["target"]
        # Use the source agent's last message as prompt (or a default if none)
        last_msg = agents[source].conversation_history[-1]["content"] if agents[
            source].conversation_history else "Hello"
        response = agents[target](last_msg)
        conversation_logs[target].append(response)
        updates.append((source, target, response))
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
                        html.P([
                            html.I(className="fas fa-info-circle me-2"),
                            "Drag nodes to reposition agents. Edges are automatically updated based on proximity."
                        ], className="text-muted fst-italic small")
                    ])
                ], className="mb-4 shadow-sm"),
                
                # Agent info panel
                dbc.Card([
                    dbc.CardHeader(html.H4("Agent Information", className="text-center")),
                    dbc.CardBody([
                        html.Div([
                            dbc.Row([
                                dbc.Col(html.Div(f"{name}", className="fw-bold"), width=4),
                                dbc.Col(html.Div(f"{agent.__class__.__name__}"), width=4),
                                dbc.Col(html.Div([
                                    html.Span(className="badge bg-warning text-dark",
                                            children=agent.model)
                                ]), width=4)
                            ], className="mb-2 node-card")
                            for name, agent in agents.items()
                        ])
                    ])
                ], className="mb-4 shadow-sm"),
                
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
    
    return html.Div([
        html.H5([
            "Conversation: ", 
            html.Span(f"{source}", className="text-warning fw-bold"), 
            " → ", 
            html.Span(f"{target}", className="text-warning fw-bold")
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
                html.Div(msg, className="p-2 mb-2 bg-light rounded")
                for msg in conversation_logs.get(target, ["No messages yet"])
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

# Add auto-update feature every 10 seconds (if desired, but commented out by default)
# app.clientside_callback(
#     """
#     function(n_intervals) {
#         // This would auto-step the simulation at a set interval
#         const btn = document.getElementById('step-btn');
#         if (btn) btn.click();
#         return window.dash_clientside.no_update;
#     }
#     """,
#     Output("cytoscape", "elements", allow_duplicate=True),
#     Input(dcc.Interval(id="auto-interval", interval=10000, disabled=True), "n_intervals"),
#     prevent_initial_call=True
# )

if __name__ == '__main__':
    app.run_server(debug=True)
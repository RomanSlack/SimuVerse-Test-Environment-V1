import streamlit as st
import os
from dotenv import load_dotenv
import math
import dash
from dash import html, dcc, Input, Output, State
import dash_cytoscape as cyto
from modules.framework import create_agent
# -------------------------
# Load API Keys and Create Agents
# -------------------------
load_dotenv()
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
app = dash.Dash(__name__)
server = app.server

app.layout = html.Div([
    html.H1("Multi-Agent Interactive Simulation"),
    cyto.Cytoscape(
        id='cytoscape',
        elements=generate_elements(agent_positions),
        style={'width': '600px', 'height': '600px'},
        layout={'name': 'preset'},
        stylesheet=[
            {'selector': 'node', 'style': {'label': 'data(label)',
                                           'background-color': '#0074D9',
                                           'width': 50, 'height': 50}},
            {'selector': 'edge', 'style': {'line-color': '#AAAAAA',
                                           'target-arrow-color': '#AAAAAA',
                                           'target-arrow-shape': 'triangle'}}
        ],
        userPanningEnabled=True,
        userZoomingEnabled=True,
        autoungrabify=False
    ),
    html.Button("Step Simulation", id="step-btn", n_clicks=0),
    html.Div(id="log-div", style={"margin-top": "20px", "whiteSpace": "pre-line"}),
    html.Div("Drag nodes to reposition them. Edges are automatically updated based on proximity.",
             style={"margin-top": "20px", "fontStyle": "italic"})
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
# Callback: Display Logs When an Edge is Clicked
# -------------------------
@app.callback(
    Output("log-div", "children"),
    Input('cytoscape', 'tapEdgeData')
)
def display_logs(edgeData):
    if edgeData is None:
        return "Click on an edge to view conversation logs."
    source = edgeData.get("source")
    target = edgeData.get("target")
    logs = f"Conversation Logs for connection {source} -> {target}:\n"
    logs += f"{source} logs:\n" + "\n".join(conversation_logs.get(source, [])) + "\n\n"
    logs += f"{target} logs:\n" + "\n".join(conversation_logs.get(target, []))
    return logs


if __name__ == '__main__':
    app.run_server(debug=True)
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import random
from mpl_toolkits.mplot3d import Axes3D  # Required for 3D plotting

# Number of agents in the simulation
num_agents = 10

# Create a directed graph where nodes represent agents
G = nx.DiGraph()

# Initialize agent positions randomly within a unit cube (3D)
positions = {i: (random.random(), random.random(), random.random()) for i in range(num_agents)}
for i in range(num_agents):
    # Each agent starts with a default message and status
    G.add_node(i, message="Hello", status="idle")

# Set up the 3D plot using a valid style
plt.style.use('seaborn-v0_8-dark-palette')
fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, projection='3d')
ax.set_facecolor('#f0f0f0')
ax.set_axis_off()  # Hide axes for a cleaner look

# Define colors for each status
status_colors = {"talking": "orange", "thinking": "purple", "idle": "skyblue"}
# Coefficient for attraction force when agents communicate
attraction_coef = 0.03


def update(frame):
    ax.clear()
    ax.set_facecolor('#f0f0f0')
    ax.set_axis_off()

    # Update positions: apply slight random jitter in 3D
    for i in range(num_agents):
        x, y, z = positions[i]
        new_x = min(max(x + random.uniform(-0.03, 0.03), 0), 1)
        new_y = min(max(y + random.uniform(-0.03, 0.03), 0), 1)
        new_z = min(max(z + random.uniform(-0.03, 0.03), 0), 1)
        positions[i] = (new_x, new_y, new_z)

    # Randomly update each agent's message and status
    for i in range(num_agents):
        if random.random() < 0.3:
            G.nodes[i]['message'] = random.choice(["Hello", "How are you?", "Yes", "No", "Maybe", "Goodbye"])
            G.nodes[i]['status'] = random.choice(["talking", "thinking", "idle"])

    # Clear existing conversation edges and create new ones randomly
    G.clear_edges()
    for i in range(num_agents):
        if random.random() < 0.5:
            target = random.randint(0, num_agents - 1)
            if target != i:
                G.add_edge(i, target)

    # Apply an attraction force for each conversation edge so that agents move closer in 3D
    for i, j in G.edges():
        xi, yi, zi = positions[i]
        xj, yj, zj = positions[j]
        dx = xj - xi
        dy = yj - yi
        dz = zj - zi
        positions[i] = (xi + attraction_coef * dx, yi + attraction_coef * dy, zi + attraction_coef * dz)
        positions[j] = (xj - attraction_coef * dx, yj - attraction_coef * dy, zj - attraction_coef * dz)

    # Draw nodes with colors based on their status and add a black outline
    for i in G.nodes():
        x, y, z = positions[i]
        ax.scatter(x, y, z, s=100, c=status_colors[G.nodes[i]['status']], edgecolors='black', depthshade=True)
        # Add text labels with a slight offset for clarity
        ax.text(x, y, z, f"Agent {i}\n{G.nodes[i]['message']}", fontsize=9, color='black')

    # Draw edges as dashed lines between agents
    for i, j in G.edges():
        xi, yi, zi = positions[i]
        xj, yj, zj = positions[j]
        ax.plot([xi, xj], [yi, yj], [zi, zj], color='gray', linewidth=1, linestyle='--')

    ax.set_title("Simuverse 3D Agent Interaction Visualization", fontsize=18)


ani = animation.FuncAnimation(fig, update, frames=200, interval=1000)
plt.show()

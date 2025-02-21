import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import random
from enum import Enum


# Number of agents in the simulation
num_agents = 5

# Coefficient for attraction force when agents communicate
attraction_coef = 0.03


class Status(Enum):
    TALKING = "orange"
    THINKING = "purple"
    IDLE = "skyblue"


def create_plot():
    # Set up the 3D plot using a valid style
    plt.style.use('seaborn-v0_8-dark-palette')
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_facecolor('#f0f0f0')
    ax.set_axis_off()  # Hide axes for a cleaner look
    return ax, fig


# Create a directed graph where nodes represent agents
G = nx.DiGraph()

# Initialize agent positions randomly within a unit cube (3D)
positions = {i: (random.random(), random.random(), random.random()) for i in range(num_agents)}

# Each agent starts with a default message and status
for i in range(num_agents):
    G.add_node(i, message="Hello", status=Status.IDLE)


def update(plot):
    plot.clear()
    plot.set_facecolor('#f0f0f0')
    plot.set_axis_off()

    # Randomly update each agent's message and status
    for i in range(num_agents):
        if random.random() < 0.3:
            G.nodes[i]['message'] = random.choice(["Hello", "How are you?", "Yes", "No", "Maybe", "Goodbye"])
            G.nodes[i]['status'] = random.choice([Status.TALKING, Status.THINKING, Status.IDLE])

    # Clear existing conversation edges and create new ones randomly
    G.clear_edges()
    for i in range(num_agents):
        if random.random() < 0.5:
            target = random.randint(0, num_agents - 1)
            if target != i:
                G.add_edge(i, target)

    # Draw nodes with colors based on their status and add a black outline
    for i in G.nodes():
        x, y, z = positions[i]
        plot.scatter(x, y, z, s=100, c=G.nodes[i]['status'].value, edgecolors='black', depthshade=True)
        plot.text(x, y, z, f"Agent {i}\n{G.nodes[i]['message']}", fontsize=9, color='black')

    # Draw edges as dashed lines between agents
    for i, j in G.edges():
        xi, yi, zi = positions[i]
        xj, yj, zj = positions[j]
        plot.plot([xi, xj], [yi, yj], [zi, zj], color='gray', linewidth=1, linestyle='--')

    plot.set_title("Simuverse 3D Agent Interaction Visualization", fontsize=18)


def main():
    plot, fig = create_plot()
    animation.FuncAnimation(fig, update(plot), frames=1, interval=1000, repeat=False)
    plt.show()


if __name__ == "__main__":
    main()

import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import random

# Coefficient for attraction force when agents communicate
attraction_coef = 0.03

# Create a directed graph where nodes represent agents
G = nx.DiGraph()


def create_plot():
    # Set up the 3D plot using a valid style
    plt.style.use('seaborn-v0_8-dark-palette')
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_facecolor('#f0f0f0')
    ax.set_axis_off()  # Hide axes for a cleaner look
    return ax, fig


def update(plot, agents, positions):
    plot.clear()
    plot.set_facecolor('#f0f0f0')
    plot.set_axis_off()

    # Clear existing conversation edges and create new ones randomly
    G.clear_edges()
    # Clear the names of each node
    for node in G.nodes():
        G.nodes[node]['message'] = None

    # Send agent info to nodes
    for i in range(len(agents)):
        msg = agents[i].last_msg
        if msg:
            # Make sure the index of agents is the same as its id-1
            G.add_edge(msg.sender_id-1, msg.recipient_id-1)
            G.nodes[i]['message'] = msg.content
        G.nodes[i]['status'] = agents[i].state

    # Draw nodes with colors based on their status and add a black outline
    for i in G.nodes():
        x, y, z = positions[i]
        plot.scatter(
            x, y, z,
            s=100,
            c=G.nodes[i]['status'].value,
            edgecolors='black',
            depthshade=True
        )

        # Display the message if it exists
        if 'message' in G.nodes[i]:
            plot.text(
                x, y, z,
                f"Agent {i}\n{G.nodes[i]['message']}",
                fontsize=9,
                color='black'
            )
        else:
            plot.text(
                x, y, z,
                f"Agent {i}", fontsize=9, color='black'
            )

    # Draw edges as dashed lines between agents
    for i, j in G.edges():
        xi, yi, zi = positions[i]
        xj, yj, zj = positions[j]
        plot.plot(
            [xi, xj], [yi, yj], [zi, zj],
            color='gray',
            linewidth=1,
            linestyle='--'
        )

    plot.set_title("Simuverse 3D Agent Interaction Visualization", fontsize=18)


def visualize(agents):
    # Initialize agent positions randomly within a unit cube (3D)
    positions = {
        i: (random.random(), random.random(), random.random())
        for i in range(len(agents))
    }

    # Create a node for each agent
    [G.add_node(i, status=None) for i in range(len(agents))]

    plot, fig = create_plot()

    # Pass the update function itself, not the result of the function call
    animation.FuncAnimation(
        fig,
        update(plot, agents, positions),  # This is the function itself
        frames=1,
        interval=1000,
        repeat=False
    )

    fig.set_size_inches(plt.figaspect(1) * fig.get_dpi() / fig.get_dpi())
    plt.show()

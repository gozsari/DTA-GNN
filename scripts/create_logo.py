import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from pathlib import Path


def create_logo(output_path):
    fig, ax = plt.subplots(figsize=(4, 4), dpi=300)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_xlim(-0.1, 1.1)
    ax.set_ylim(0, 1.0)

    # Colors
    hex_color = "#4F46E5"  # Indigo
    node_color = "#EC4899"  # Pink
    edge_color = "#6366F1"  # Lighter Indigo

    # 1. Draw a benzene-like hexagon (Drug)
    hex_center = (0.25, 0.6)
    hex_radius = 0.18
    theta = np.linspace(0, 2 * np.pi, 7) + np.pi / 6  # rotated
    hex_x = hex_center[0] + hex_radius * np.cos(theta)
    hex_y = hex_center[1] + hex_radius * np.sin(theta)

    # Outer hexagon
    poly = patches.Polygon(
        np.column_stack([hex_x, hex_y]),
        closed=True,
        facecolor="none",
        edgecolor=hex_color,
        linewidth=5,
        joinstyle="round",
    )
    ax.add_patch(poly)

    # Inner circle to suggest aromaticity
    circle = patches.Circle(
        hex_center,
        radius=hex_radius * 0.6,
        facecolor="none",
        edgecolor=hex_color,
        linewidth=2,
        linestyle="--",
    )
    ax.add_patch(circle)

    # 2. Draw Network Nodes (Target/interaction)
    # Positions relative to hexagon
    nodes = [(0.65, 0.75), (0.65, 0.45), (0.90, 0.60)]

    # Edges linking Hexagon to Network
    # Connect from a vertex of hexagon (right-most vertex roughly, index 6)
    hex_vertex = (
        hex_center[0] + hex_radius * np.cos(theta[6]),
        hex_center[1] + hex_radius * np.sin(theta[6]),
    )

    # Draw edges
    for node in nodes:
        # Edge from molecule to nodes
        ax.plot(
            [hex_vertex[0], node[0]],
            [hex_vertex[1], node[1]],
            color=edge_color,
            linewidth=2,
            zorder=1,
            alpha=0.6,
        )

    # Inter-node edges
    ax.plot(
        [nodes[0][0], nodes[1][0]],
        [nodes[0][1], nodes[1][1]],
        color=edge_color,
        linewidth=2,
        zorder=1,
        alpha=0.6,
    )
    ax.plot(
        [nodes[1][0], nodes[2][0]],
        [nodes[1][1], nodes[2][1]],
        color=edge_color,
        linewidth=2,
        zorder=1,
        alpha=0.6,
    )
    ax.plot(
        [nodes[0][0], nodes[2][0]],
        [nodes[0][1], nodes[2][1]],
        color=edge_color,
        linewidth=2,
        zorder=1,
        alpha=0.6,
    )

    # Draw Nodes
    for node in nodes:
        n = patches.Circle(
            node,
            radius=0.04,
            facecolor=node_color,
            edgecolor="white",
            linewidth=2,
            zorder=2,
        )
        ax.add_patch(n)

    # 3. Text
    ax.text(
        0.5,
        0.2,
        "DTA-GNN",
        fontsize=28,
        ha="center",
        va="center",
        fontfamily="sans-serif",
        fontweight="bold",
        color="#1F2937",
    )

    ax.text(
        0.5,
        0.1,
        "Dataset Builder",
        fontsize=12,
        ha="center",
        va="center",
        fontfamily="sans-serif",
        color="#6B7280",
    )

    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_file, transparent=True, bbox_inches="tight", pad_inches=0.1)
    print(f"Logo saved to {output_file}")


if __name__ == "__main__":
    create_logo("assets/logo.png")

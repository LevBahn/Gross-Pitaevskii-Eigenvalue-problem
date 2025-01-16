import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Initialize figure
fig, ax = plt.subplots(figsize=(12, 6))

# Neural network layers
input_layer = [-3, -1, 1, 3]
hidden_layer_1 = [-4, -2, 0, 2, 4]
hidden_layer_2 = [-4, -2, 0, 2, 4]
output_layer = [0]

# Positions for layers
x_positions = [0, 1.5, 3, 4.5]
layers = [input_layer, hidden_layer_1, hidden_layer_2, output_layer]

# Draw nodes
for x, layer in zip(x_positions, layers):
    for y in layer:
        ax.add_patch(patches.Circle((x, y), 0.1, color="lightblue", ec="black", zorder=2))

# Draw connections
for i, (x1, layer1) in enumerate(zip(x_positions[:-1], layers[:-1])):
    x2 = x_positions[i + 1]
    layer2 = layers[i + 1]
    for y1 in layer1:
        for y2 in layer2:
            ax.plot([x1, x2], [y1, y2], "k-", lw=0.5, alpha=0.7, zorder=1)

# Annotations for losses and components
ax.text(5.5, 2, "Losses", fontsize=14, bbox=dict(boxstyle="round", fc="white", ec="black"))
ax.text(-1, 2, "Input", fontsize=14, bbox=dict(boxstyle="round", fc="white", ec="black"))
ax.text(4.5, -4, "Output", fontsize=14, bbox=dict(boxstyle="round", fc="white", ec="black"))

# Add arrows for losses
arrow_start = (4.5, 2.5)
arrow_end = (6, 2.5)
ax.annotate("", xy=arrow_end, xytext=arrow_start, arrowprops=dict(facecolor="black", arrowstyle="->"))

# Final adjustments
ax.set_xlim(-2, 7)
ax.set_ylim(-5, 5)
ax.axis("off")
plt.show()

import matplotlib.pyplot as plt
import numpy as np
from tueplots.constants.color import palettes, rgb


def plot_tue_colors():
    """Displays all colors from tueplots.constants.color.rgb with their names."""

    # Extract color names and values while ensuring we only get valid colors
    color_names = []
    colors = []

    for attr in dir(rgb):
        if not attr.startswith("__") and not callable(
            getattr(rgb, attr)
        ):  # Ignore dunder methods and callables
            color_value = getattr(rgb, attr)
            if isinstance(color_value, np.ndarray) and color_value.shape == (
                3,
            ):  # Ensure it's an RGB array
                color_names.append(attr)
                colors.append(color_value)

    # Ensure all colors are within the [0,1] range
    colors = [color / 255.0 if np.max(color) > 1 else color for color in colors]

    # Adjust figure size for better readability
    fig_width = max(
        10, max(len(name) for name in color_names) * 0.4
    )  # Scale width based on longest name
    fig_height = len(colors) * 0.5
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    ax.set_xlim(0, 1)
    ax.set_ylim(0, len(colors))

    # Plot each color with its name
    for i, (name, color) in enumerate(zip(color_names, colors)):
        ax.add_patch(plt.Rectangle((0, i), 1, 1, color=color))
        ax.text(1.05, i + 0.5, name, va="center", ha="left", fontsize=12)

    # Remove axes and adjust layout
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_frame_on(False)
    plt.subplots_adjust(left=0.1, right=0.8)  # Add space for text labels

    # Show the color chart
    # plt.savefig("utils/plots/tue_colors.png")
    plt.show()


def plot_tue_palettes():
    """Displays all palettes from tueplots.constants.color.palettes with their names."""

    # Extract palette names and values
    palette_names = []
    palette_colors = []

    for attr in dir(palettes):
        if not attr.startswith("__") and not callable(
            getattr(palettes, attr)
        ):  # Ignore dunder methods and callables
            palette_value = getattr(palettes, attr)
            if (
                isinstance(palette_value, np.ndarray) and len(palette_value.shape) == 2
            ):  # Ensure it's an array of colors
                palette_names.append(attr)
                palette_colors.append(palette_value)

    # Normalize color values to [0,1] range if necessary
    for i, colors in enumerate(palette_colors):
        if np.max(colors) > 1:
            palette_colors[i] = colors / 255.0

    # Adjust figure size dynamically
    fig_width = 10
    fig_height = len(palette_names) * 1.5
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    ax.set_xlim(
        0, len(max(palette_colors, key=len))
    )  # Adjust width based on the longest palette
    ax.set_ylim(0, len(palette_names))

    # Plot each palette with its name
    for i, (name, colors) in enumerate(zip(palette_names, palette_colors)):
        for j, color in enumerate(colors):
            ax.add_patch(plt.Rectangle((j, i), 1, 1, color=color))
        ax.text(len(colors) + 0.5, i + 0.5, name, va="center", ha="left", fontsize=12)

    # Remove axes and adjust layout
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_frame_on(False)
    plt.subplots_adjust(left=0.1, right=0.8)  # Add space for text labels

    # Show the color palettes
    # plt.savefig("utils/plots/tue_palettes.png")
    plt.show()


# Example usage:
if __name__ == "__main__":
    plot_tue_colors()
    plot_tue_palettes()

import os

import matplotlib.pyplot as plt
import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from tueplots.constants.color import palettes, rgb


def extract_tensorboard_data(log_dir, metric="eval/mean_reward", smooth_factor=0.9):
    """
    Extracts and smooths data from TensorBoard event files.

    Args:
        log_dir (str): Parent directory containing experiment subdirectories.
        metric (str): The metric to extract from TensorBoard logs.
        smooth_factor (float): The smoothing factor for moving average (0 = no smoothing, 1 = full smooth).

    Returns:
        dict: A dictionary with experiment names as keys and (steps, values) as values.
    """
    experiment_data = {}

    for experiment in sorted(os.listdir(log_dir)):
        experiment_path = os.path.join(log_dir, experiment)
        if os.path.isdir(experiment_path) and experiment.startswith("PPO_"):
            event_file = None
            for file in os.listdir(experiment_path):
                if file.startswith("events.out.tfevents"):
                    event_file = os.path.join(experiment_path, file)
                    break

            if event_file:
                event_acc = EventAccumulator(event_file, size_guidance={"scalars": 0})
                event_acc.Reload()

                if metric in event_acc.Tags()["scalars"]:
                    steps, values = [], []
                    for scalar_event in event_acc.Scalars(metric):
                        steps.append(scalar_event.step)
                        values.append(scalar_event.value)

                    if steps and values:
                        # Apply exponential moving average smoothing
                        smoothed_values = []
                        last_value = values[0]
                        for v in values:
                            last_value = last_value * smooth_factor + v * (
                                1 - smooth_factor
                            )
                            smoothed_values.append(last_value)

                        experiment_data[experiment] = (steps, smoothed_values)

    return experiment_data


def plot_tensorboard_data(
    log_dir,
    metric="eval/mean_reward",
    smooth_factor=0.9,
    custom_styles=None,
    xlabel="Training Steps",
    ylabel="Mean Reward",
    title="Training Progress",
    ylim=None,  # Limit Y-axis
    save_path=None,  # Path to save the plot
    figsize=(10, 6),  # New: Set figure size
):
    """
    Extracts and plots smoothed TensorBoard data.

    Args:
        log_dir (str): Parent directory containing experiment subdirectories.
        metric (str): The metric to extract and plot.
        smooth_factor (float): The smoothing factor for moving average (0 = no smoothing, 1 = full smooth).
        custom_styles (dict): Custom colors and labels, e.g., {"PPO_12": ("blue", "Experiment A")}.
        xlabel (str): Label for the x-axis.
        ylabel (str): Label for the y-axis.
        title (str): Title of the plot.
        ylim (tuple): (y_min, y_max) to restrict the Y-axis range.
        save_path (str): If provided, saves the plot to this path instead of displaying it.
        figsize (tuple): Size of the figure in inches (width, height).
    """
    data = extract_tensorboard_data(log_dir, metric, smooth_factor)

    plt.figure(figsize=figsize)  # Use custom figure size

    for experiment, (steps, values) in data.items():
        color, label = custom_styles.get(
            experiment, (None, experiment)
        )  # Default label = folder name
        plt.plot(steps, values, label=label, color=color)

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)

    if ylim:
        plt.ylim(ylim)  # Apply Y-axis limits if provided

    # plt.legend(ncol=2, loc="lower right")
    plt.legend()
    plt.grid()

    # Save the plot if a path is provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Plot saved to {save_path}")
    else:
        plt.show()


# Example usage with custom colors, labels, axis titles, Y-axis limits, and saving
custom_styles_noises = {
    "PPO_12": (palettes.tue_plot[-1], "Brown"),
    "PPO_13": (palettes.tue_plot[0], "No noise"),
    "PPO_14": (palettes.tue_plot[-3], "Pink"),
    "PPO_15": (palettes.tue_plot[4], "White"),
    "PPO_16": (palettes.tue_plot[3], "Gaussian"),
    "PPO_17": (palettes.tue_plot[1], "Ornstein"),
}

custom_styles_rewards = {
    "PPO_0": (rgb.tue_ocre, "1-puck_infront"),
    "PPO_1": (rgb.tue_lightorange, "1-pred_dist_from_puck"),
    "PPO_2": (rgb.tue_red, "10-puck_infront"),
    "PPO_3": (rgb.tue_orange, "10-pred_dist_from_puck"),
    "PPO_4": (rgb.tue_darkblue, "10-puck_positional"),  #
    "PPO_5": (rgb.tue_green, "10-blocking"),
    "PPO_6": (rgb.tue_lightgreen, "1-blocking"),
    "PPO_7": (rgb.tue_blue, "1-puck_positional"),  #
    "PPO_8": (rgb.tue_dark, "10-defensive_play"),  #
    "PPO_9": (rgb.tue_brown, "10-momentum_control"),  #
    "PPO_10": (rgb.tue_gold, "1-momentum_control"),  #
    "PPO_11": (rgb.tue_gray, "1-defensive_play"),  #
}

custom_styles_rew_mult = {
    "PPO_0": (palettes.tue_plot[-1], "5"),
    "PPO_1": (palettes.tue_plot[0], "1"),
    "PPO_2": (palettes.tue_plot[-3], "10"),
    "PPO_3": (palettes.tue_plot[4], "2"),
    "PPO_4": (palettes.tue_plot[1], "20"),
}

custom_styles_rewards3= {
    "PPO_0": (palettes.tue_plot[-1], "momentum_control"),
    "PPO_1": (palettes.tue_plot[0], "defensive_play"),
    "PPO_2": (palettes.tue_plot[-3], "puck_positional"),
}

if __name__ == "__main__":

    # plot_tensorboard_data(
    #     log_dir="/storage/brno2/home/nademvit/rl_hw/rl-hockey/ppo/models/noises",
    #     smooth_factor=0.95,
    #     custom_styles=custom_styles_noises,
    #     xlabel="Number of Training Steps",
    #     ylabel="Smoothed Mean Reward",
    #     title="Effect of Different Noise Types on PPO Performance",
    #     ylim=(-2.5, 10),  # Restrict Y-axis range
    #     save_path="utils/plots/ppo_noises.png",  # Save the plot instead of showing it
    #     figsize=(12, 4),
    # )

    # plot_tensorboard_data(
    #     log_dir="/storage/brno2/home/nademvit/rl_hw/rl-hockey/ppo/models/rewards",
    #     smooth_factor=0.95,
    #     custom_styles=custom_styles_rewards,
    #     xlabel="Number of Training Steps",
    #     ylabel="Smoothed Mean Reward",
    #     title="Effect of Different Reward Types on PPO Performance",
    #     ylim=(-25, 10),  # Restrict Y-axis range
    #     save_path="utils/plots/ppo_rewards.png",  # Save the plot instead of showing it
    #     figsize=(12, 6),
    # )
    
    # plot_tensorboard_data(
    #     log_dir="/storage/brno2/home/nademvit/rl_hw/rl-hockey/ppo/models/rew_mult",
    #     smooth_factor=0.95,
    #     custom_styles=custom_styles_rew_mult,
    #     xlabel="Number of Training Steps",
    #     ylabel="Smoothed Mean Reward",
    #     title="Effect of Different Reward Multipliers on PPO Performance",
        # ylim=(-20, 10),  
    #     save_path="utils/plots/ppo_rew_mult.png",  # Save the plot instead of showing it
    #     figsize=(12, 4),
    # )
    
    plot_tensorboard_data(
        log_dir="/storage/brno2/home/nademvit/rl_hw/rl-hockey/ppo/logs",
        smooth_factor=0.95,
        custom_styles=custom_styles_rewards3,
        xlabel="Number of Training Steps",
        ylabel="Smoothed Mean Reward",
        title="Effect of Different Rewards3 on PPO Performance",
        # ylim=(-20, 10),  
        save_path="utils/plots/ppo_rewards3.png",  # Save the plot instead of showing it
        figsize=(12, 4),
    )

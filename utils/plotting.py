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
    xlim=None,  # Limit X-axis
    ylim=None,  # Limit Y-axis
    save_path=None,  # Path to save the plot
    figsize=(10, 6),  # Set figure size
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
    
    # Print best eval/mean_reward for each experiment
    print("\nBest eval/mean_reward for each experiment:")
    for experiment, (steps, values) in data.items():
        best_idx = np.argmax(values)
        best_value = values[best_idx]
        best_step = steps[best_idx]
        best_name = custom_styles.get(experiment, experiment)[1]
        print(f"  {experiment}[{best_name}]: Best {metric} = {best_value:.2f} at step {best_step:,}")

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
    
    if xlim:
        plt.xlim(xlim)  # Apply X-axis limits if provided

    # plt.legend(ncol=3, loc="lower right")
    plt.legend()
    plt.grid()

    # Save the plot if a path is provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Plot saved to {save_path}")
    else:
        plt.show()

def sort_key(label):
    if label == "vanilla":
        return (1000, 1000)  # Ensure "vanilla" is sorted last
    parts = label.split("_")
    e_val = int(parts[0][1:])  # Extract numeric part after 'e'
    i_val = float(parts[1][1:])  # Extract numeric part after 'i'
    return (e_val, i_val)  # Sort first by eX, then by iY


def plot_tensorboard_data_RND(
    log_dir,
    metric="eval/mean_reward",
    smooth_factor=0.9,
    custom_styles=None,
    xlabel="Training Steps",
    ylabel="Mean Reward",
    title="Training Progress",
    xlim=None,  
    ylim=None,  
    save_path=None,  
    figsize=(10, 6),
):
    """
    Extracts and plots smoothed TensorBoard data.

    Args:
        log_dir (str): Parent directory containing experiment subdirectories.
        metric (str): The metric to extract and plot.
        smooth_factor (float): The smoothing factor for moving average (0 = no smoothing, 1 = full smooth).
        custom_styles (dict): Custom colors and labels.
        xlabel (str): Label for the x-axis.
        ylabel (str): Label for the y-axis.
        title (str): Title of the plot.
        ylim (tuple): (y_min, y_max) to restrict the Y-axis range.
        save_path (str): If provided, saves the plot to this path instead of displaying it.
        figsize (tuple): Size of the figure in inches (width, height).
    """
    data = extract_tensorboard_data(log_dir, metric, smooth_factor)
    
    # Print best eval/mean_reward for each experiment
    print("\nBest eval/mean_reward for each experiment:")
    for experiment, (steps, values) in data.items():
        best_idx = np.argmax(values)
        best_value = values[best_idx]
        best_step = steps[best_idx]
        best_name = custom_styles.get(experiment, experiment)[1]
        print(f"  {experiment}[{best_name}]: Best {metric} = {best_value:.2f} at step {best_step:,}")

    plt.figure(figsize=figsize)  

    for experiment, (steps, values) in data.items():
        color, label = custom_styles.get(
            experiment, (None, experiment)
        )  
        plt.plot(steps, values, label=label, color=color)

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)

    if ylim:
        plt.ylim(ylim)  
    
    if xlim:
        plt.xlim(xlim)  

    # Extract current legend handles and labels
    handles, labels = plt.gca().get_legend_handles_labels()

    # Separate "vanilla" from other labels
    vanilla_handle = None
    vanilla_label = None
    filtered_handles = []
    filtered_labels = []

    for handle, label in zip(handles, labels):
        if label == "vanilla":
            vanilla_handle = handle
            vanilla_label = label
        else:
            filtered_handles.append(handle)
            filtered_labels.append(label)

    # Sort numerically: first by eX, then by iY
    def sort_key(label):
        if label == "vanilla":
            return (1000, 1000)  # Ensure "vanilla" is sorted last
        parts = label.split("_")
        e_val = int(parts[0][1:])  # Extract numeric part after 'e'
        i_val = float(parts[1][1:])  # Extract numeric part after 'i'
        return (e_val, i_val)  # Sort first by eX, then by iY

    sorted_pairs = sorted(zip(filtered_handles, filtered_labels), key=lambda x: sort_key(x[1]))
    sorted_handles, sorted_labels = zip(*sorted_pairs)

    # Place all entries except 'vanilla' in a multi-column legend
    main_legend = plt.legend(sorted_handles, sorted_labels, ncol=3, loc="lower right")

    # Add 'vanilla' separately in its own row
    if vanilla_handle:
        vanilla_legend = plt.legend([vanilla_handle], [vanilla_label], loc="center right", bbox_to_anchor=(0.975, 0.36))

    plt.gca().add_artist(main_legend)  # Keep both legends active
    plt.grid()

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
    "vanilla": (rgb.mps_green, "nothing"),  #
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
    "PPO_5": (palettes.tue_plot[4], "blocking"),
    "vanilla": (palettes.tue_plot[5], "nothing"),
}

custom_styles_rewards4= {
    "PPO_2": (palettes.tue_plot[0], "momentum_control"),
    "PPO_0": (palettes.tue_plot[1], "puck_proximity"),
    "PPO_1": (palettes.tue_plot[2], "intercept_path"),
    "PPO_3": (palettes.tue_plot[3], "offensive_pressure"),
    "PPO_4": (palettes.tue_plot[4], "positional_control"),
    "PPO_5": (palettes.tue_plot[5], "defensive_coverage"),
    "PPO_vanilla": (palettes.tue_plot[6], "vanilla"),
    "PPO_blocking": (palettes.tue_plot[7], "blocking"),
    "PPO_momentum_control": (palettes.tue_plot[8], "momentum_control"),
}

custom_styles_rewards5= {
    "PPO_0": (rgb.tue_darkgreen, "puck_positional"),
    "PPO_1": (palettes.tue_plot[1], "defensive_play"),
    "PPO_2": (palettes.tue_plot[3], "momentum_control"),
    "PPO_3": (palettes.tue_plot[8], "offensive_pressure"),
    "PPO_4": (palettes.tue_plot[4], "puck_between_player_and_goal"),
    "PPO_5": (palettes.tue_plot[5], "blocking"),
    "PPO_6": (palettes.tue_plot[6], "intercept_path"),
    "PPO_7": (palettes.tue_plot[7], "puck_proximity"),
    "PPO_vanilla": (palettes.tue_plot[0], "vanilla"),
}

custom_styles_rnd = {
    "PPO_1": (palettes.tue_secondary[1], "e0_i0.01"),
    "PPO_0": (palettes.tue_secondary[0], "e0_i0.1"),
    "PPO_2": (palettes.tue_secondary[2], "e0_i1"),  
    "PPO_3": (palettes.tue_secondary[3], "e0_i10"),
    "PPO_vanilla": (palettes.tue_primary[0], "vanilla"),
    
    # "PPO_4": (palettes.tue_secondary[4], "e1_i0.01"),
    "PPO_15": (palettes.tue_secondary[4], "e1_i0.01"),
    # "PPO_6": (palettes.tue_secondary[5], "e1_i0.1"),
    "PPO_13": (palettes.tue_secondary[5], "e1_i0.1"),
    # "PPO_5": (palettes.tue_secondary[6], "e1_i1"),
    "PPO_14": (palettes.tue_secondary[6], "e1_i1"),
    "PPO_9": (palettes.tue_secondary[7], "e1_i10"),
    
    "PPO_8": (palettes.tue_secondary[8], "e10_i0.01"),
    "PPO_11": (palettes.tue_secondary[11], "e10_i0.1"),
    "PPO_10": (palettes.tue_secondary[10], "e10_i1"),
    "PPO_7": (palettes.tue_secondary[9], "e10_i10"),   
}

custom_styles_threerews = {
    "PPO_0": (palettes.tue_plot[-1], "ip+pp+op"),
    "PPO_1": (palettes.tue_plot[0], "pp+op_G"),
    "PPO_2": (palettes.tue_plot[-3], "pp+op"),
    "PPO_3": (palettes.tue_plot[4], "pp_G"),
    "PPO_4": (palettes.tue_plot[1], "op_G"),
}

custom_styles_rew_mult2 = {
    "PPO_5": (palettes.tue_plot[2], "r1"),
    "PPO_6": (palettes.tue_plot[3], "r2"),
    "PPO_8": (palettes.tue_plot[6], "r5"),
    "PPO_7": (palettes.tue_plot[5], "r10"),
    "PPO_9": (palettes.tue_plot[7], "r15"),
    "PPO_10": (palettes.tue_plot[8], "r20"),
}

custom_styles_rnd_parallel = {
    "PPO_0": (palettes.tue_plot[0], "64"),
    "PPO_1": (palettes.tue_plot[1], "128"),
    "PPO_2": (palettes.tue_plot[2], "32"),
    "PPO_3": (palettes.tue_plot[3], "256"),
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
    
    # plot_tensorboard_data(
    #     log_dir="/storage/brno2/home/nademvit/rl_hw/rl-hockey/ppo/models/rewards3",
    #     smooth_factor=0.95,
    #     custom_styles=custom_styles_rewards3,
    #     xlabel="Number of Training Steps",
    #     ylabel="Smoothed Mean Reward",
    #     title="Effect of Different Rewards3 on PPO Performance",
    #     # ylim=(-20, 10),  
    #     save_path="utils/plots/ppo_rewards3.png",  # Save the plot instead of showing it
    #     figsize=(12, 4),
    # )
    
    
    
    # plot_tensorboard_data(
    #     log_dir="/storage/brno2/home/nademvit/rl_hw/rl-hockey/ppo/logs/rnd_parallel",
    #     smooth_factor=0.95,
    #     custom_styles={},
    #     xlabel="Number of Training Steps",
    #     ylabel="Smoothed Mean Reward",
    #     title="Effect of n_envs on PPO+RND Performance",
    #     # ylim=(-15, 12),  
    #     # xlim=(0, 2e7),
    #     save_path="utils/plots/ppo_rnd_nenvs.png",  # Save the plot instead of showing it
    #     figsize=(12, 4),
    # )
    
    # plot_tensorboard_data(
    #     log_dir="/storage/brno2/home/nademvit/rl_hw/rl-hockey/ppo/logs/noises2",
    #     smooth_factor=0.95,
    #     custom_styles={},
    #     xlabel="Number of Training Steps",
    #     ylabel="Smoothed Mean Reward",
    #     title="Effect of Gaussian noise on PPO Performance",
    #     # ylim=(-15, 12),  
    #     # xlim=(0, 2e7),
    #     save_path="utils/plots/ppo_gauss.png",  # Save the plot instead of showing it
    #     figsize=(12, 4),
    # )
    
    # plot_tensorboard_data(
    #     log_dir="/storage/brno2/home/nademvit/rl_hw/rl-hockey/ppo/logs/rewards5",
    #     smooth_factor=0.95,
    #     custom_styles=custom_styles_rewards5,
    #     xlabel="Number of Training Steps",
    #     ylabel="Smoothed Mean Reward",
    #     title="Effect of Different Rewards on PPO Performance",
    #     ylim=(-10, 12),  
    #     # xlim=(0, 2e7),
    #     save_path="utils/plots/ppo_rewards5.png",  # Save the plot instead of showing it
    #     figsize=(12, 4),
    # )
    
    # plot_tensorboard_data_RND(
    #     log_dir="/storage/brno2/home/nademvit/rl_hw/rl-hockey/ppo/logs/rnd",
    #     smooth_factor=0.95,
    #     custom_styles=custom_styles_rnd,
    #     xlabel="Number of Training Steps",
    #     ylabel="Smoothed Mean Reward",
    #     title="Effect of RND on PPO Performance",
    #     # ylim=(-20, 12),  
    #     # xlim=(2e7, 3e7),
    #     save_path="utils/plots/ppo_rnd.png",  # Save the plot instead of showing it
    #     figsize=(12, 4),
    # )

    # plot_tensorboard_data(
    #     log_dir="/storage/brno2/home/nademvit/rl_hw/rl-hockey/ppo/logs/threerews",
    #     smooth_factor=0.95,
    #     custom_styles=custom_styles_threerews,
    #     xlabel="Number of Training Steps",
    #     ylabel="Smoothed Mean Reward",
    #     title="Effect of Multiple Reward and Noise Combination on PPO Performance",
    #     # ylim=(-15, 12),  
    #     # xlim=(0, 2e7),
    #     save_path="utils/plots/ppo_3rews.png",  # Save the plot instead of showing it
    #     figsize=(12, 4),
    # )
    
    # plot_tensorboard_data(
    #     log_dir="/storage/brno2/home/nademvit/rl_hw/rl-hockey/ppo/logs/rew_mult2",
    #     smooth_factor=0.95,
    #     custom_styles=custom_styles_rew_mult2,
    #     xlabel="Number of Training Steps",
    #     ylabel="Smoothed Mean Reward",
    #     title="Effect of Reward Multiplier on PPO Performance",
    #     # ylim=(-15, 12),  
    #     # xlim=(0, 2e7),
    #     save_path="utils/plots/ppo_rew_mult2.png",  # Save the plot instead of showing it
    #     figsize=(12, 4),
    # )
    
    # plot_tensorboard_data(
    #     log_dir="/storage/brno2/home/nademvit/rl_hw/rl-hockey/ppo/logs/rnd_parallel",
    #     smooth_factor=0.95,
    #     custom_styles=custom_styles_rnd_parallel,
    #     xlabel="Number of Training Steps",
    #     ylabel="Smoothed Mean Reward",
    #     title="Effect of Number of Parallel Environments on PPO RND Performance",
    #     # ylim=(-15, 12),  
    #     # xlim=(0, 2e7),
    #     save_path="utils/plots/ppo_rnd_parallel.png",  # Save the plot instead of showing it
    #     figsize=(12, 4),
    # )

    plot_tensorboard_data(
        log_dir="/storage/brno2/home/nademvit/rl_hw/rl-hockey/ppo/logs/weak",
        smooth_factor=0.95,
        custom_styles=custom_styles_rewards4,
        xlabel="Number of Training Steps",
        ylabel="Smoothed Mean Reward",
        title="Effect of Number of TEMP on PPO RND Performance",
        # ylim=(-15, 12),  
        # xlim=(0, 2e7),
        save_path="utils/plots/ppo_temp.png",  # Save the plot instead of showing it
        figsize=(12, 4),
    )
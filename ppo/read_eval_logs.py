import argparse

import matplotlib.pyplot as plt
import numpy as np


def parse_args():
    """
    Parse command-line arguments for processing and visualizing EvalCallback .npz logs.

    Returns:
        argparse.Namespace: Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Parse and visualize EvalCallback .npz logs."
    )
    parser.add_argument(
        "--npz_path",
        type=str,
        required=True,
        help="Path to the .npz file containing EvalCallback logs.",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="ppo/eval_results.png",
        help="Path to save the output plot image.",
    )
    return parser.parse_args()


def process_npz(npz_path, output_path):
    """
    Load and process the .npz file containing evaluation results, then plot and save the results.

    Args:
        npz_path (str): Path to the .npz file.
        output_path (str): Path to save the output plot image.
    """
    # Load the npz file
    data = np.load(npz_path)

    # Inspect available keys
    print("Keys:", data.files)

    # Extract data
    timesteps = data["timesteps"]
    results = data["results"]

    # Compute mean rewards if results are multidimensional
    mean_rewards = results.mean(axis=1) if results.ndim > 1 else results

    # Display results in the terminal (optional)
    # for t, r in zip(timesteps, mean_rewards):
    # print(f"Timestep: {t}, Mean Reward: {r}")

    # Plot the data and save as an image
    plt.plot(timesteps, mean_rewards)
    plt.xlabel("Timesteps")
    plt.ylabel("Mean Reward")
    plt.title("Evaluation Results")
    plt.savefig(output_path)
    print(f"Plot saved to {output_path}")


def main():
    """
    Main function to parse arguments and process the provided .npz file.
    """
    args = parse_args()
    process_npz(args.npz_path, args.output_path)


if __name__ == "__main__":
    main()

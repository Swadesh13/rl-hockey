import os


def find_highest_numbered_folder(base_path, prefix="PPO_"):
    """
    Find the folder with the highest number in its name (format: PPO_<number>) in a given directory.

    :param base_path: The path to the directory containing the folders.
    :return: The path to the folder with the highest number or None if no matching folder is found.
    """
    highest_number = -1
    highest_folder_path = None

    for folder_name in os.listdir(base_path):
        if folder_name.startswith(prefix) and folder_name[4:].isdigit():
            number = int(folder_name[4:])
            if number > highest_number:
                highest_number = number
                highest_folder_path = os.path.join(base_path, folder_name)

    return highest_folder_path


# Example usage:
if __name__ == "__main__":
    base_directory = "/home/vojta/Documents/rl-hockey/ppo/ppo_hockey_tensorboard"  # Replace with your folder path
    highest_folder = find_highest_numbered_folder(base_directory)
    if highest_folder:
        print(f"The folder with the highest number is: {highest_folder}")
    else:
        print("No matching folder found.")

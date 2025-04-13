""" 
This script visualizes the distribution of data in .npy files.
It loads .npy files from a specified directory and prints the shape of the data.

Then it visualizes the data using matplotlib.
Files are structured as root_dir/scene_name/asset.npy
    where asset can be coord, color, normal, segment.

The distribution of the data is calculated per each scene_name.

"""

import os
import numpy as np
import matplotlib.pyplot as plt
import argparse
import glob

def analyze_scene(directory, output_dir):
    scene_name = os.path.basename(directory)
    # Get all .npy files in the directory
    npy_files = glob.glob(os.path.join(directory, '*.npy'))
    print("Analyzing scene:", directory)
    # Load each .npy file and store its data
    for npy_file in npy_files:
        # Get the base name of the file (e.g., coord.npy)
        base_name = os.path.basename(npy_file)
        # Load the .npy file
        data = np.load(npy_file)
        # Print the shape of the data
        if base_name == "coord.npy":
            print(f"File: {base_name}, Shape: {data.shape}")
        elif base_name == "color.npy":
            # Color data (r,g,b) analyze with histogram
            plt.figure(figsize=(10, 5))
            plt.hist(data[:, 0], bins=50, alpha=0.5, label='Red')
            plt.hist(data[:, 1], bins=50, alpha=0.5, label='Green')
            plt.hist(data[:, 2], bins=50, alpha=0.5, label='Blue')
            plt.title(f"Color Distribution in {base_name}")
            plt.xlabel('Color Value')
            plt.ylabel('Frequency')
            plt.legend()
            # Save the histogram
            plt.savefig(os.path.join(output_dir, f"{scene_name}-color_histogram.png"))
            plt.close()
        elif base_name == "segment.npy":
            # This class contains the class ids, analyze distribution of classes
            unique, counts = np.unique(data, return_counts=True)
            plt.figure(figsize=(10, 5))
            plt.bar(unique, counts)
            plt.title(f"Segment Distribution in {base_name}")
            plt.xlabel('Class ID')
            plt.ylabel('Frequency')
            # Save the histogram
            plt.savefig(os.path.join(output_dir, f"{scene_name}-classes_bar.png"))
            plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize .npy files")
    parser.add_argument("--npy_dir", type=str, help="Directory containing .npy files")
    parser.add_argument("--output_dir", type=str, help="Directory to save the output files")
    args = parser.parse_args()

    # Assert that the output directory exists
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # Get all the directories in the specified directory
    directories = [d for d in os.listdir(args.npy_dir) if os.path.isdir(os.path.join(args.npy_dir, d))]
    # For each directory, analyze the scene
    for directory in directories:
        scene_path = os.path.join(args.npy_dir, directory)
        analyze_scene(scene_path, args.output_dir)

# Visualize some npy files head and tail

import os
import numpy as np
import matplotlib.pyplot as plt
import argparse
import glob

def visualize_npy_files(npy_dir):
    # Get all .npy files in the directory and its subdirectories
    npy_files = glob.glob(os.path.join(npy_dir, '**', '*.npy'), recursive=True)
    # Get all .npy files in the directory
    #npy_files = glob.glob(os.path.join(npy_dir, '*.npy'))
    sorted_len = []
    names = []
    for npy_file in npy_files:
        # for now print only the one with base name coord.npy
        if os.path.basename(npy_file) != "coord.npy":
            continue
        # Load the .npy file
        data = np.load(npy_file)
        """# Print the first and last 5 elements of the data
        print(f"File: {npy_file}")
        print("First 5 elements:", data[:5])
        print("Last 5 elements:", data[-5:])
        print("Shape:", data.shape)
        print("-" * 40)  """
        # Store the length of the data
        sorted_len.append(data.shape[0])
        names.append(npy_file)
    # Sort the lengths
    indexes = np.argsort(sorted_len)
    sorted_len = np.array(sorted_len)[indexes]
    names = np.array(names)[indexes]
    # Print the sorted names
    print("Sorted names of coord.npy files:")
    for name, lenght in zip(names, sorted_len):
        print(name, lenght)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize .npy files")
    parser.add_argument("npy_dir", type=str, help="Directory containing .npy files")
    args = parser.parse_args()

    # Visualize the .npy files in the specified directory
    visualize_npy_files(args.npy_dir)

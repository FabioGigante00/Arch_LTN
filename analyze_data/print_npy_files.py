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

    for npy_file in npy_files:
        # for now print only the one with base name coord.npy
        if os.path.basename(npy_file) != "coords.npy":
            continue
        # Load the .npy file
        data = np.load(npy_file)
        # Print the first and last 5 elements of the data
        print(f"File: {npy_file}")
        #print("First 5 elements:", data[:5])
        #print("Last 5 elements:", data[-5:])
        print("Shape:", data.shape)
        print("-" * 40) 

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize .npy files")
    parser.add_argument("npy_dir", type=str, help="Directory containing .npy files")
    args = parser.parse_args()

    # Visualize the .npy files in the specified directory
    visualize_npy_files(args.npy_dir)

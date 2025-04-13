"""
Preprocessing Script for ArchDataset
Normal vectors are already provided in the dataset.

The single row of the txt file is structured as:
[x, y, z, r, g, b, class_id, Nx, Ny, Nz]

Classes:
- "arch": 0
- "column": 1
- "moldings": 2
- "floor": 3
- "door_window": 4
- "wall": 5
- "stairs": 6
- "vault": 7
- "roof": 8
- "other": 9

"""

import os
import argparse
import glob
import numpy as np

from concurrent.futures import ProcessPoolExecutor
from itertools import repeat

def parse_scene(
    scene, dataset_root, output_root
):
    print("Parsing: {}".format(scene))
    classes = [
        "arch",
        "column",
        "moldings",
        "floor",
        "door_window",
        "wall",
        "stairs",
        "vault",
        "roof",
        "other"
    ]
    class2label = {cls: i for i, cls in enumerate(classes)}
    source_path = os.path.join(dataset_root, scene)
    # Remove .txt from the scene name
    scene = os.path.splitext(scene)[0]
    save_path = os.path.join(output_root, scene)
    os.makedirs(save_path, exist_ok=True)

    scene_name = os.path.basename(source_path)
    obj = np.loadtxt(source_path)
    coords = obj[:, :3]
    colors = obj[:, 3:6]
    class_id = obj[:, 6]
    class_id = class_id.astype(np.int32).reshape([-1, 1])
    normals = obj[:, 7:10]

    print("Processing scene: ", scene_name)
    print("Shape of coords: ", coords.shape)
    print("Shape of colors: ", colors.shape)
    print("Shape of normals: ", normals.shape)
    print("Shape of class_id: ", class_id.shape)
    print("Save path: ", save_path)
    # Save the processed data in save_path
    coords_path = os.path.join(save_path, "coord.npy")
    colors_path = os.path.join(save_path, "color.npy")
    normals_path = os.path.join(save_path, "normal.npy")
    segment_path = os.path.join(save_path, "segment.npy")
    np.save(coords_path, coords)
    np.save(colors_path, colors)
    np.save(normals_path, normals)
    np.save(segment_path, class_id)




def main_process():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_root", type=str, help="Path to Arch Dataset"
    )
    parser.add_argument(
        "--output_root", type=str, help="Path to save processed data"
    )

    args = parser.parse_args()
    if args.dataset_root is None:
        raise ValueError("Please provide the dataset root path.")
    if args.output_root is None:
        # Construct the output path based on the dataset root
        # Root + 'processed'
        args.output_root = os.path.join(args.dataset_root, "processed")
    
    scene_training = []
    scene_testing = []

    # Check if the dataset root exists
    if not os.path.exists(args.dataset_root):
        raise ValueError("Dataset root path does not exist.")
    # Create the output directory adding base name of dataset root
    basename = os.path.basename(args.dataset_root)
    args.output_root = os.path.join(args.output_root, basename)
    # Create the output directory if it doesn't exist
    os.makedirs(args.output_root, exist_ok=True)
    print("Output root path: ", args.output_root)
    # Load the scene list from the dataset root
    print("Loading training scene list...")
    training_root = os.path.join(args.dataset_root, "Training")
    testing_root = os.path.join(args.dataset_root, "Test")
    scene_training = glob.glob(os.path.join(training_root, "*.txt"))
    scene_training_names = [os.path.basename(scene) for scene in scene_training]
    print("Training scene list loaded.")
    print("Loading testing scene list...")
    scene_testing = glob.glob(os.path.join(testing_root, "*.txt"))
    scene_testing_names = [os.path.basename(scene) for scene in scene_testing]
    print("Testing scene list loaded.")

    # Create the output directory if it doesn't exist
    os.makedirs(args.output_root, exist_ok=True)
    # Process the training scenes
    print("Processing training scenes...")
    # For now not using multiprocessing

    for scene in scene_training_names:
        output_dir = os.path.join(args.output_root, 'Training')
        # Create the output directory for training scenes
        os.makedirs(output_dir, exist_ok=True)
        # Parse the scene and save the data
        parse_scene(scene, training_root, output_dir)
    print("Training scenes processed.")
    # Process the testing scenes
    print("Processing testing scenes...")
    for scene in scene_testing_names:
        output_dir = os.path.join(args.output_root, 'Test')
        # Create the output directory for testing scenes
        os.makedirs(output_dir, exist_ok=True)
        # Parse the scene and save the data
        parse_scene(scene, testing_root, output_dir)
    print("Testing scenes processed.")
    print("All scenes processed.")
    print("Done.")


if __name__ == "__main__":
    main_process()
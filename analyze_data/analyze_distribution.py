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
from scipy.spatial import ConvexHull
from scipy.spatial import cKDTree
import open3d as o3d
from collections import defaultdict
from matplotlib import cm

class_names = [
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

def analyze_scene(directory, output_dir, create_all_charts, show_pc_density):
    text_output = ''
    scene_name = os.path.basename(directory)
    # Get all .npy files in the directory
    npy_files = glob.glob(os.path.join(directory, '*.npy'))
    print("Analyzing scene:", directory)
    text_output += f'Scene {scene_name}\n'
    # Load each .npy file and store its data
    for npy_file in npy_files:
        # Get the base name of the file (e.g., coord.npy)
        base_name = os.path.basename(npy_file)
        # Load the .npy file
        data = np.load(npy_file)
        # Print the shape of the data
        if base_name == "coord.npy":
            #print(f"File: {base_name}, Shape: {data.shape}")
            text_output += f"Number of points: {data.shape}\n"
            density = round(estimate_density(data), 2)
            text_output += f"Average density: {density} points per cubic unit \n"

            # Calculate min and max z, min and max y, and min and max x to analyze scene scale
            min_x, max_x = np.min(data[:, 0]), np.max(data[:, 0])
            min_y, max_y = np.min(data[:, 1]), np.max(data[:, 1])
            min_z, max_z = np.min(data[:, 2]), np.max(data[:, 2])

            depth = max_y - min_y
            width = max_x - min_x
            height = max_z - min_z
            text_output += f"Scene dimensions (depth, width, height): {depth}, {width}, {height}\n"

            if not show_pc_density:
                continue
            spacing = (1.0 / density) ** (1/3)
            radius = 1.5 * spacing  # to smooth over a bit larger neighborhood
            color_points_by_density(data, radius)

        elif base_name == "color.npy":
            if not create_all_charts:
                continue
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
            save_dir = os.path.join(output_dir, "charts", "color", f"{scene_name}-color_histogram.png")
            os.makedirs(os.path.dirname(save_dir), exist_ok=True)
            plt.savefig(save_dir)
            plt.close()
        elif base_name == "segment.npy":
            # This class contains the class ids, analyze distribution of classes
            unique, counts = np.unique(data, return_counts=True)
            percentage = counts / np.sum(counts)
            assert(np.sum(counts) == data.shape[0])
            percentage = [f"{round(p * 100, 2)}%" for p in percentage]
            #print(f"{unique}\n {counts}\n {percentage}\n")
            max_class_name_length = max(len(name) for name in class_names)
            for i in range(len(unique)):
                text_output += f"{class_names[unique[i]].ljust(max_class_name_length)}: \t{percentage[i].ljust(5)}\t({counts[i]})\n"

            if not create_all_charts:
                continue
            plt.figure(figsize=(10, 5))
            plt.bar(class_names[unique], counts)
            plt.title(f"Segment Distribution in {base_name}")
            plt.xlabel('Class ID')
            plt.ylabel('Frequency')
            # Save the histogram
            save_dir = os.path.join(output_dir, "charts", "segment", f"{scene_name}-classes_bar.png")
            os.makedirs(os.path.dirname(save_dir), exist_ok=True)
            plt.savefig(save_dir)
            plt.close()
    return text_output, unique, counts


def estimate_density(points):
    """
    Estimate point cloud density as average points per unit volume.
    
    Args:
        points (np.ndarray): Nx3 array of XYZ coordinates.
    
    Returns:
        float: Estimated density.
    """
    if points.shape[1] != 3:
        raise ValueError("Points must be a Nx3 array of XYZ coordinates.")
    
    hull = ConvexHull(points)
    volume = hull.volume
    num_points = points.shape[0]
    
    density = num_points / volume if volume > 0 else 0
    return density

def estimate_local_density(points, radius=0.1):
    tree = cKDTree(points)
    densities = []
    for pt in points:
        idx = tree.query_ball_point(pt, radius)
        volume = (4/3) * np.pi * (radius ** 3)
        densities.append(len(idx) / volume)
    return np.array(densities)

def color_points_by_density(points, radius=0.1):
    """
    Color each point by the number of neighbors within `radius`.
    """
    # 1) Build Open3D PointCloud and KD‐tree
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    kdtree = o3d.geometry.KDTreeFlann(pcd)

    # 2) Compute density for each point
    densities = np.zeros(len(points), dtype=np.int32)
    for i, pt in enumerate(pcd.points):
        # search all points within `radius`
        _, idxs, _ = kdtree.search_radius_vector_3d(pt, radius)
        densities[i] = len(idxs)

    # 3) Normalize densities to [0,1]
    dens_f = densities.astype(float)
    dens_f = (dens_f - dens_f.min()) / (dens_f.ptp() + 1e-6)

    # 4) Map to colors via a colormap
    cmap = plt.colormaps.get_cmap('Reds')
    colors = cmap(dens_f)[:, :3]   # drop alpha

    # 5) Assign colors and visualize
    pcd.colors = o3d.utility.Vector3dVector(colors)
    o3d.visualization.draw_geometries([pcd])



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize .npy files")
    parser.add_argument("--npy_dir", type=str, help="Directory containing .npy files")
    parser.add_argument("--output_dir", type=str, help="Directory to save the output files")
    parser.add_argument("--create_all_charts", type=str, help="flag for creating charts", default="no")
    parser.add_argument("--show_pc_density", type=str, help="flag for showing the point clouds by density", default="no")
    args = parser.parse_args()

    # Assert that the output directory exists
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    
    # Determine if charts should be created
    create_all_charts = args.create_all_charts.strip().lower() == "yes"
    show_pc_density = args.show_pc_density.strip().lower() == "yes"

    # Get all the directories in the specified directory
    directories = [d for d in os.listdir(args.npy_dir) if os.path.isdir(os.path.join(args.npy_dir, d))]
    # For each directory, analyze the scene
    txt = ''
    split_agg = []
    split_names = []
    split_agg_percentages = []
    train_scenes = 0
    for n_split, split in enumerate(directories):
        split_path = os.path.join(args.npy_dir, split)
        scenes = [d for d in os.listdir(split_path) if os.path.isdir(os.path.join(split_path, d))]
        txt += f'{split} split\n'
        split_dict = {i: 0 for i in range(10)}

        if os.path.basename(split).lower() == "training":
            train_scenes = len(scenes)
        print(os.path.basename(split).lower())

        for scene in scenes:
            scene_path = os.path.join(split_path, scene)
            add_txt, unique, counts = analyze_scene(scene_path, args.output_dir, create_all_charts, show_pc_density)
            txt += add_txt + '\n\n'
            for index, i in enumerate(unique):
                split_dict[i] += counts[index]
        split_percentages = [0] * len(split_dict)
        for i in range(len(split_dict)):
            split_percentages[i] = split_dict[i] / sum(split_dict.values())
        split_percentages = [f"{round(p * 100, 2)}%" for p in split_percentages]

        txt += "Class Distribution:\n"
        max_class_name_length = max(len(name) for name in class_names)
        for i in range(len(class_names)):
            txt += f"{class_names[i].ljust(max_class_name_length)}: \t{split_percentages[i].ljust(5)}\t({split_dict[i]})\n"
        txt += '\n\n'

        # Add in split_agg[n_split], the number of points for each class
        split_agg.append(split_dict)
        split_agg_percentages.append(split_percentages)
        split_names.append(split)
    
    # Create a bar chart for the n. of points of each class for each split
    plt.figure(figsize=(15, 7))
    bar_width = 0.2
    x = np.arange(len(class_names))

    for i in range(len(split_agg)):
        plt.bar(x + i * bar_width, [split_agg[i][j] for j in range(10)], bar_width, label=split_names[i])

    plt.title("Class Distribution in each split")
    plt.xlabel('Class ID')
    plt.ylabel('Frequency')
    plt.xticks(x + bar_width * (len(split_agg) - 1) / 2, class_names, rotation=45)
    plt.legend()
    # Save the histogram
    save_dir = os.path.join(args.output_dir, "charts", "split_classes_bar.png")
    os.makedirs(os.path.dirname(save_dir), exist_ok=True)
    plt.savefig(save_dir)
    plt.close()

    # Create a bar chart for the n. of points of each class for each split
    # For the Train split, use the average number of points per class
    plt.figure(figsize=(15, 7))
    bar_width = 0.2
    x = np.arange(len(class_names))

    print("Number of Train scenes", train_scenes)
    for i in range(len(split_agg)):
        if split_names[i].lower() == "training":
            avg_points = {j: split_agg[i][j] / train_scenes for j in range(10)}
            plt.bar(x + i * bar_width, [avg_points[j] for j in range(10)], bar_width, label=f"{split_names[i]} (avg)")
        else:
            plt.bar(x + i * bar_width, [split_agg[i][j] for j in range(10)], bar_width, label=split_names[i])

    plt.title("Class Distribution in each split (Train uses average points)")
    plt.xlabel('Class ID')
    plt.ylabel('Frequency')
    plt.xticks(x + bar_width * (len(split_agg) - 1) / 2, class_names, rotation=45)
    plt.legend()
    # Save the histogram
    save_dir = os.path.join(args.output_dir, "charts", "split_classes_bar_avg_train.png")
    os.makedirs(os.path.dirname(save_dir), exist_ok=True)
    plt.savefig(save_dir)
    plt.close()

    # Create a bar chart for the percentage of points of each class for each split
    plt.figure(figsize=(15, 7))
    bar_width = 0.2
    x = np.arange(len(class_names))

    for i in range(len(split_agg_percentages)):
        split_percentages = [float(p.strip('%')) / 100 for p in split_agg_percentages[i]]
        plt.bar(x + i * bar_width, split_percentages, bar_width, label=split_names[i])

    plt.title("Class Distribution Percentages in each split")
    plt.xlabel('Class ID')
    plt.ylabel('Percentage')
    plt.xticks(x + bar_width * (len(split_agg_percentages) - 1) / 2, class_names, rotation=45)
    plt.legend()
    # Save the histogram
    save_dir = os.path.join(args.output_dir, "charts", "split_classes_percentage_bar.png")
    os.makedirs(os.path.dirname(save_dir), exist_ok=True)
    plt.savefig(save_dir)
    plt.close()

    # Save file 
    os.makedirs(os.path.join(args.output_dir, "info"), exist_ok=True)
    with open(os.path.join(args.output_dir, "info", "distribution.txt"), 'w') as f:
        f.write(txt)
# split_pointcloud.py

import os
import sys
import numpy as np

def split_arrays(arrays, max_points):
    """
    Split multiple aligned arrays into chunks of max_points.
    Args:
        arrays: list of np.ndarray, all with same first dimension (N).
        max_points: int, max number of points per chunk.
    Returns:
        list of tuple of arrays: [(arr1_chunk, arr2_chunk, ...), ...]
    """
    n_points = arrays[0].shape[0]
    chunks = []
    
    for start_idx in range(0, n_points, max_points):
        end_idx = min(start_idx + max_points, n_points)
        chunk = tuple(arr[start_idx:end_idx] for arr in arrays)
        chunks.append(chunk)
    
    return chunks

def main(input_folder, max_points):
    # Load the arrays
    coord = np.load(os.path.join(input_folder, 'coord.npy'))
    color = np.load(os.path.join(input_folder, 'color.npy'))
    normal = np.load(os.path.join(input_folder, 'normal.npy'))
    segment = np.load(os.path.join(input_folder, 'segment.npy'))

    arrays = [coord, color, normal, segment]

    # Check all arrays have the same number of points
    n_points = coord.shape[0]
    for arr in arrays:
        if arr.shape[0] != n_points:
            raise ValueError("All input arrays must have the same number of points!")

    # Split
    parts = split_arrays(arrays, max_points)

    # Output
    base_name = os.path.basename(input_folder.rstrip('/'))

    for i, part in enumerate(parts):
        output_dir = f"{base_name}_Part{i}"
        os.makedirs(output_dir, exist_ok=True)

        np.save(os.path.join(output_dir, 'coord.npy'), part[0])
        np.save(os.path.join(output_dir, 'color.npy'), part[1])
        np.save(os.path.join(output_dir, 'normal.npy'), part[2])
        np.save(os.path.join(output_dir, 'segment.npy'), part[3])

        print(f"Saved {output_dir}/ with {part[0].shape[0]} points.")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python split_pointcloud.py <InputFolder> <MaxPoints>")
        sys.exit(1)
    
    input_folder = sys.argv[1]
    max_points = int(sys.argv[2])

    main(input_folder, max_points)

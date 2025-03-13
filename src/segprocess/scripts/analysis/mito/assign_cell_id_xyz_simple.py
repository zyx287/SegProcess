'''
author: zyx
date: 2025-02-10
last modified: 2025-03-12
description: 
    Step3: Assign cell id to each mitochondria, works for xyz order data
    This is a new version
'''

import os
import pickle
import numpy as np
import trimesh
import multiprocessing
from tqdm import tqdm
import zarr

def process_mito_label(mito_label, resolution=16, num_samples=10):
    """
    Processes a single mitochondrial label using multiple vertex samples.
    
    Args:
        mito_label: The mitochondria label to process
        resolution: Voxel resolution for converting vertex coordinates to voxel coordinates (default: 16)
        num_samples: Number of vertices to sample (default: 10)
        
    Returns:
        Tuple of (mito_label, cell_id) or None if processing failed
    """
    ply_path = os.path.join(mito_ply_dir, f"{mito_label}_mito_s1.ply")

    if not os.path.exists(ply_path):
        print(f"PLY file not found for mito {mito_label}.")
        return None

    try:
        mesh = trimesh.load_mesh(ply_path)
        
        # Get total number of vertices
        total_vertices = len(mesh.vertices)
        
        # Ensure we don't try to sample more vertices than exist
        effective_samples = min(num_samples, total_vertices)
        
        # Randomly sample vertex indices
        if total_vertices > effective_samples:
            sample_indices = np.random.choice(total_vertices, effective_samples, replace=False)
            sample_vertices = mesh.vertices[sample_indices]
        else:
            # If we have fewer vertices than requested samples, use all vertices
            sample_vertices = mesh.vertices
        
        # Track cell IDs for each sampled vertex
        cell_id_votes = {}
        valid_samples = 0
        
        for vertex in sample_vertices:
            # Convert vertex to downsampled voxel coordinates
            vertex_voxel = np.floor(np.round(vertex) / resolution).astype(int)
            
            # Check if within bounds
            if (0 <= vertex_voxel[0] < cell_segmentation.shape[0] and
                0 <= vertex_voxel[1] < cell_segmentation.shape[1] and
                0 <= vertex_voxel[2] < cell_segmentation.shape[2]):
                
                # Get cell ID at vertex location
                cell_id = cell_segmentation[vertex_voxel[0], vertex_voxel[1], vertex_voxel[2]]
                
                # Only count non-zero cell IDs
                if cell_id != 0:
                    valid_samples += 1
                    cell_id_votes[cell_id] = cell_id_votes.get(cell_id, 0) + 1
        
        # If no valid samples found, try the centroid as fallback
        if valid_samples == 0:
            centroid = mesh.centroid
            centroid_voxel = np.floor(np.round(centroid) / 16).astype(int)
            
            # Check if within bounds
            if (0 <= centroid_voxel[0] < cell_segmentation.shape[0] and
                0 <= centroid_voxel[1] < cell_segmentation.shape[1] and
                0 <= centroid_voxel[2] < cell_segmentation.shape[2]):
                
                cell_id = cell_segmentation[centroid_voxel[0], centroid_voxel[1], centroid_voxel[2]]
                if cell_id != 0:
                    return mito_label, cell_id
            
            print(f"No valid cell ID found for mito {mito_label} from any sample points.")
            return None
        
        # Find the most common cell ID (majority vote)
        max_votes = 0
        winning_cell_id = None
        
        for cell_id, votes in cell_id_votes.items():
            if votes > max_votes:
                max_votes = votes
                winning_cell_id = cell_id
        
        # Calculate confidence (percentage of valid samples that voted for the winning cell)
        confidence = max_votes / valid_samples
        
        # Log result with confidence
        print(f"Mito {mito_label} assigned to cell {winning_cell_id} with {confidence:.1%} confidence ({max_votes}/{valid_samples} points)")
        
        return mito_label, winning_cell_id

    except Exception as e:
        print(f"Error processing mito {mito_label}: {e}")
        return None

def process_mito_label_simple(mito_label, resolution=16):
    """
    Simplified version that processes a mitochondrial label using only the first vertex.
    
    Args:
        mito_label: The mitochondria label to process
        resolution: Voxel resolution for converting vertex coordinates (default: 16)
        
    Returns:
        Tuple of (mito_label, cell_id) or None if processing failed
    """
    ply_path = os.path.join(mito_ply_dir, f"{mito_label}_mito_s1.ply")

    if not os.path.exists(ply_path):
        return None

    try:
        mesh = trimesh.load_mesh(ply_path)
        
        # Check if mesh has vertices
        if len(mesh.vertices) == 0:
            return None
            
        # Get first vertex
        vertex = mesh.vertices[0]
        
        # Convert vertex to downsampled voxel coordinates
        vertex_voxel = np.floor(np.round(vertex) / resolution).astype(int)
        
        # Check if within bounds
        if (0 <= vertex_voxel[0] < cell_segmentation.shape[0] and
            0 <= vertex_voxel[1] < cell_segmentation.shape[1] and
            0 <= vertex_voxel[2] < cell_segmentation.shape[2]):
            
            # Get cell ID at vertex location
            cell_id = cell_segmentation[vertex_voxel[0], vertex_voxel[1], vertex_voxel[2]]
            
            # Only accept non-zero cell IDs
            if cell_id != 0:
                return mito_label, cell_id
        
        # # If first vertex didn't work, try the centroid as fallback
        # centroid = mesh.centroid
        # centroid_voxel = np.floor(np.round(centroid) / resolution).astype(int)
        
        # # Check if within bounds
        # if (0 <= centroid_voxel[0] < cell_segmentation.shape[0] and
        #     0 <= centroid_voxel[1] < cell_segmentation.shape[1] and
        #     0 <= centroid_voxel[2] < cell_segmentation.shape[2]):
            
        #     cell_id = cell_segmentation[centroid_voxel[0], centroid_voxel[1], centroid_voxel[2]]
        #     if cell_id != 0:
        #         return mito_label, cell_id
        print(f"No valid cell ID for mito {mito_label} at centroid {vertex_voxel}.")
        return None

    except Exception as e:
        print(f"Error processing mito {mito_label}: {e}")
        return None


def parallel_mapping(use_simple=True, resolution=16, num_samples=10):
    """
    Processes all mitochondrial meshes in parallel to map mito_label -> cell_id.
    
    Args:
        use_simple: Whether to use the simplified (faster) version (default: True)
        resolution: Voxel resolution for mapping (default: 16)
        num_samples: Number of samples to use if not using simple version (default: 10)
    """
    mito_to_cell = {}  # Dictionary to store mappings
    tasks = []
    pool = multiprocessing.Pool(processes=num_workers)

    print(f"Mapping mitochondria to cells using {'simple first-vertex' if use_simple else f'{num_samples} sample points'} method...")
    
    for mito_label in tqdm(mito_labels):
        if use_simple:
            tasks.append(pool.apply_async(process_mito_label_simple, args=(mito_label, resolution)))
        else:
            tasks.append(pool.apply_async(process_mito_label, args=(mito_label, resolution, num_samples)))

    pool.close()
    pool.join()

    # Collect results
    for task in tasks:
        result = task.get()
        if result:
            mito_label, cell_id = result
            mito_to_cell[mito_label] = cell_id

    # Save mapping
    with open(mito_to_cell_path, "wb") as f:
        pickle.dump(mito_to_cell, f)

    print(f"Mitochondria-to-cell mapping saved: {len(mito_to_cell)} entries.")
    return mito_to_cell


def build_reverse_mapping(mito_to_cell):
    """
    Converts mito_label -> cell_id mapping into cell_id -> list of mito_labels.
    """
    cell_to_mito = {}  # Dictionary to store cell -> list of mitochondria

    print("Building reverse mapping (cell -> mitochondria)...")
    for mito_label, cell_id in tqdm(mito_to_cell.items()):
        if cell_id not in cell_to_mito:
            cell_to_mito[cell_id] = []
        cell_to_mito[cell_id].append(mito_label)

    # Save the reverse mapping
    with open(cell_to_mito_path, "wb") as f:
        pickle.dump(cell_to_mito, f)

    print(f"Cell-to-mitochondria mapping saved: {len(cell_to_mito)} cells with mitochondria.")
    return cell_to_mito


if __name__ == "__main__":
    # Step 1: Map mitochondria to cells
    # Define paths
    mito_ply_dir = "/media/zhangy8/ca0155b9-932b-4491-8ea9-d40a586475cf/SegmentationData/Organelle/Mitochondria/Mesh/ply/"
    cell_segmentation_path = "/media/zhangy8/ca0155b9-932b-4491-8ea9-d40a586475cf/SegmentationData/CellShape/S1/cellshape_s1_filtered_20250305_xyz_new.zarr"# this datasety is zyx order
    mito_to_cell_path = "/home/zhangy8@hhmi.org/data1/Experiment/20250304_reconstruction_test/output/mito_to_cell_20250305_V3.pkl"
    cell_to_mito_path = "/home/zhangy8@hhmi.org/data1/Experiment/20250304_reconstruction_test/output/cell_to_mito_20250305_V3.pkl"

    with open("/home/zhangy8@hhmi.org/data1/20250210_assign_mitochondria/Output/final/unique_labels.pkl", "rb") as f:
        mito_labels = pickle.load(f)

    cell_segmentation = zarr.open(cell_segmentation_path, mode='r')

    num_workers = 80  # Adjust based on your system
    use_simple_method = True
    resolution = 16
    
    # Step 1: Map mitochondria to cells
    mito_to_cell = parallel_mapping(use_simple=use_simple_method, resolution=resolution)

    # Step 2: Reverse mapping if step 1 was successful
    if mito_to_cell:
        build_reverse_mapping(mito_to_cell)
    else:
        print("Mapping aborted due to errors.")

'''
author: zyx
date: 2025-02-28
last_modified: 2025-02-28
description: 
    Visualize a zarr dataset with Neuroglancer locally, without loading the whole dataset into RAM
Useage:
    python start_neuroglancer.py </path_to_zarr>
'''

import neuroglancer
import zarr
import numpy as np
import webbrowser
import os
import argparse
import signal
import sys
import threading
import time

class ZarrAdapter:
    """
    Adapter class to make zarr arrays compatible with neuroglancer's chunked data access.
    """
    def __init__(self, zarr_array):
        self.zarr_array = zarr_array
        self.shape = zarr_array.shape
        self.dtype = zarr_array.dtype
        self.data = self  # Self-reference for Neuroglancer compatibility
        
    def __getitem__(self, key):
        """Implements chunked access for neuroglancer's data fetching"""
        return self.zarr_array[key]

def explore_zarr(zarr_path):
    """
    Explore the structure of a zarr dataset and return available arrays.
    """
    z = zarr.open(zarr_path, mode='r')
    
    arrays = []
    
    def process_group(group, path=""):
        for key in group.keys():
            item_path = f"{path}/{key}" if path else key
            if isinstance(group[key], zarr.core.Array):
                arrays.append({
                    "path": item_path,
                    "shape": group[key].shape,
                    "dtype": str(group[key].dtype),
                    "chunks": group[key].chunks
                })
            elif isinstance(group[key], zarr.hierarchy.Group):
                process_group(group[key], item_path)
    
    if isinstance(z, zarr.hierarchy.Group):
        process_group(z)
    elif isinstance(z, zarr.core.Array):
        arrays.append({
            "path": "",
            "shape": z.shape,
            "dtype": str(z.dtype),
            "chunks": z.chunks
        })
    
    return arrays

def select_array(arrays):
    """Let the user select which array to visualize."""
    if not arrays:
        print("No arrays found in the Zarr dataset")
        return None
    
    print("\nAvailable arrays:")
    for i, arr in enumerate(arrays):
        print(f"{i}: {arr['path']} - Shape: {arr['shape']}, Dtype: {arr['dtype']}, Chunks: {arr['chunks']}")
    
    if len(arrays) == 1:
        print("\nOnly one array available, selecting it automatically.")
        return arrays[0]['path']
    
    choice = input("\nSelect array number to visualize (or press Enter for 0): ")
    if not choice.strip():
        choice = 0
    else:
        try:
            choice = int(choice)
            if choice < 0 or choice >= len(arrays):
                print(f"Invalid choice. Using default (0).")
                choice = 0
        except ValueError:
            print(f"Invalid input. Using default (0).")
            choice = 0
    
    return arrays[choice]['path']

def get_array_from_path(zarr_path, array_path):
    """Get the array object from a path in the zarr dataset."""
    z = zarr.open(zarr_path, mode='r')
    
    if not array_path:
        return z
    
    parts = array_path.split('/')
    current = z
    
    for part in parts:
        if part:  # Skip empty parts from leading/trailing slashes
            current = current[part]
    
    return current

def add_seg_layer(state, name, data, voxel_size):
    """
    Add a segmentation layer to the Neuroglancer state.
    Using the pattern from the provided example code.
    
    Args:
        state: The viewer state
        name: Name of the layer
        data: The data array/adapter
        voxel_size: Voxel size as [x, y, z]
    """
    state.layers[name] = neuroglancer.SegmentationLayer(
        source=neuroglancer.LocalVolume(
            data=data,
            dimensions=neuroglancer.CoordinateSpace(
                names=["z", "y", "x"],
                units=["nm", "nm", "nm"],
                scales=voxel_size,
            ),
        ),
    )

def add_scalar_layer(state, name, data, voxel_size):
    """
    Add a scalar (image) layer to the Neuroglancer state.
    Using the pattern from the provided example code.
    
    Args:
        state: The viewer state
        name: Name of the layer
        data: The data array/adapter
        voxel_size: Voxel size as [x, y, z]
    """
    state.layers[name] = neuroglancer.ImageLayer(
        source=neuroglancer.LocalVolume(
            data=data,
            dimensions=neuroglancer.CoordinateSpace(
                names=["z", "y", "x"],
                units=["nm", "nm", "nm"],
                scales=voxel_size,
            ),
        ),
    )

def view_zarr_with_neuroglancer(zarr_path, array_path=None, voxel_size=None, 
                                bind_address='127.0.0.1', bind_port=0):
    """
    View a zarr dataset with Neuroglancer locally, without loading the whole dataset.
    
    Args:
        zarr_path: Path to the zarr dataset
        array_path: Path to a specific array within the zarr dataset
        voxel_size: List of voxel sizes for each dimension [z, y, x]
        bind_address: Address to bind the neuroglancer server to
        bind_port: Port to bind the neuroglancer server to
    """
    # Setup signal handlers for clean exit
    signal.signal(signal.SIGINT, lambda sig, frame: sys.exit(0))
    
    # Initialize neuroglancer (following example code pattern)
    neuroglancer.set_server_bind_address(bind_address=bind_address, bind_port=bind_port)
    viewer = neuroglancer.Viewer()
    
    # If no array path is specified, explore the zarr dataset and let user select
    if array_path is None:
        print("Exploring Zarr dataset structure...")
        arrays = explore_zarr(zarr_path)
        array_path = select_array(arrays)
        if array_path is None:
            return
    
    # Get the array (but don't load it into memory)
    array = get_array_from_path(zarr_path, array_path)
    print(f"\nSelected array: {array_path}")
    print(f"Shape: {array.shape}, Dtype: {array.dtype}, Chunks: {array.chunks}")
    
    # Create adapter for accessing zarr data
    zarr_adapter = ZarrAdapter(array)
    
    # Set default voxel size if not provided - default to [1, 1, 1] for 3D
    if voxel_size is None:
        if len(array.shape) >= 3:
            voxel_size = [1, 1, 1]  # Default to isotropic voxels
        else:
            voxel_size = [1] * len(array.shape)
    
    # Adjust voxel_size if it doesn't match the dimensionality
    if len(voxel_size) < 3 and len(array.shape) >= 3:
        # Pad with 1's if not enough values provided
        voxel_size = voxel_size + [1] * (3 - len(voxel_size))
    
    # Set up the Neuroglancer viewer using the transaction pattern
    with viewer.txn() as state:
        # Configure basic settings
        state.showSlices = False  # Hide the 2D slice views by default
        
        # Determine if segmentation or raw data based on dtype
        layer_name = os.path.basename(array_path) if array_path else "data"
        
        # Segmentation layers for integer data (except uint8 which is often used for images)
        if np.issubdtype(array.dtype, np.integer) and array.dtype != np.uint8:
            add_seg_layer(state, layer_name, zarr_adapter, voxel_size)
        else:
            # Image layers for floating point and uint8 data
            add_scalar_layer(state, layer_name, zarr_adapter, voxel_size)
    
    # Open the neuroglancer viewer in a browser
    viewer_url = viewer.get_viewer_url()
    print(f"\nNeuroglancer viewer ready at: {viewer_url}")
    webbrowser.open(viewer_url)
    
    print("\nServer running. Press Ctrl+C to exit.")
    
    # Keep the server running until interrupted
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nShutting down...")
        sys.exit(0)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="View a zarr dataset with Neuroglancer locally")
    parser.add_argument("zarr_path", help="Path to the zarr dataset")
    parser.add_argument("--array", help="Path to a specific array within the zarr dataset")
    parser.add_argument("--voxel-size", help="Comma-separated voxel sizes (e.g., '1,1,1')")
    parser.add_argument("--bind-address", default="127.0.0.1", help="Address to bind the neuroglancer server to")
    parser.add_argument("--bind-port", type=int, default=0, help="Port to bind the neuroglancer server to")
    
    args = parser.parse_args()
    
    # Check if the path exists
    if not os.path.exists(args.zarr_path):
        print(f"Error: The path {args.zarr_path} does not exist")
        sys.exit(1)
    
    # Parse voxel size if provided
    voxel_size = None
    if args.voxel_size:
        voxel_size = [float(x) for x in args.voxel_size.split(',')]
    
    view_zarr_with_neuroglancer(
        args.zarr_path, 
        args.array, 
        voxel_size, 
        args.bind_address, 
        args.bind_port
    )
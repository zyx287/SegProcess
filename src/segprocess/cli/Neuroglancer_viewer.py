'''
author: zyx
date: 2025-03-01
last_modified: 2025-03-01
description: 
    Use segorocess.visualization.start_neuroglancer to recall neuroglancer for visualizing zarr dataset
'''

from segprocess.visualization.start_neuroglancer import (
    view_zarr_with_neuroglancer,
    explore_zarr,
    add_seg_layer,
    add_scalar_layer
)
import argparse
import pathlib

def analyze_zarr_structure(zarr_path):
    # Get information about all arrays in the zarr dataset
    arrays = explore_zarr(zarr_path)
    
    # Print summary
    print(f"Found {len(arrays)} arrays in zarr dataset")
    for i, arr in enumerate(arrays):
        print(f"Array {i}: {arr['path']}")
        print(f"  Shape: {arr['shape']}")
        print(f"  Dtype: {arr['dtype']}")
        print(f"  Chunks: {arr['chunks']}")
    
    return arrays

def main():
    parser = argparse.ArgumentParser(description="segprocess zarr viewer")
    parser.add_argument("zarr_path", default=None,
                         help="Path to the zarr dataset")
    parser.add_argument("--array", default=None,
                        help="Path to a specific array within the zarr dataset")
    parser.add_argument("--voxel_size", default=None,
                        help="Comma-separated voxel sizes (e.g., '1,1,1')")

    args = parser.parse_args()

    # check zarr exists
    zarr_path = pathlib.Path(args.zarr_path)
    if not zarr_path.exists():
        raise FileNotFoundError(f"Zarr dataset not found: {args.zarr_path}")

    if args.array is None:
        # If no array is specified, print a summary of the zarr dataset
        info_array = analyze_zarr_structure(args.zarr_path)
    else:
        info_array = args.array


    if args.voxel_size:
        voxel_size = [float(x) for x in args.voxel_size.split(',')]
    else:
        voxel_size = None
    
    view_zarr_with_neuroglancer(args.zarr_path, args.array, voxel_size)

if __name__ == "__main__":
    main()
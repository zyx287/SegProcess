"""
Command-line interface for visualizing zarr datasets with Neuroglancer.
"""

import argparse
import pathlib
import sys
from typing import List, Optional

from segprocess.visualization.neuroglancer import (
    view_zarr_with_neuroglancer,
    explore_zarr
)

def add_arguments(parser: argparse.ArgumentParser) -> None:
    """
    Add arguments to the parser.
    
    Args:
        parser: ArgumentParser instance
    """
    parser.add_argument("zarr_path", 
                        help="Path to the zarr dataset")
    parser.add_argument("--array", 
                        help="Path to a specific array within the zarr dataset")
    parser.add_argument("--voxel-size", 
                        help="Comma-separated voxel sizes (e.g., '1,1,1')")
    parser.add_argument("--bind-address", 
                        default="127.0.0.1", 
                        help="Address to bind the neuroglancer server to")
    parser.add_argument("--bind-port", 
                        type=int, 
                        default=8080, 
                        help="Port to bind the neuroglancer server to")
    parser.add_argument("--no-browser", 
                        action="store_true", 
                        help="Don't open a browser window automatically")

def analyze_zarr_structure(zarr_path: str) -> List[dict]:
    """
    Get information about all arrays in the zarr dataset.
    
    Args:
        zarr_path: Path to the zarr dataset
        
    Returns:
        List of dictionaries containing array metadata
    """
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

def main(args: argparse.Namespace) -> int:
    """
    Main entry point for the neuroglancer viewer.
    
    Args:
        args: Command-line arguments
        
    Returns:
        Exit code
    """
    # Check zarr exists
    zarr_path = pathlib.Path(args.zarr_path)
    if not zarr_path.exists():
        print(f"Error: Zarr dataset not found: {args.zarr_path}", file=sys.stderr)
        return 1

    # If no array is specified, print a summary of the zarr dataset
    if args.array is None:
        try:
            analyze_zarr_structure(args.zarr_path)
        except Exception as e:
            print(f"Error analyzing zarr structure: {e}", file=sys.stderr)
            return 1
    
    # Parse voxel size if provided
    voxel_size = None
    if args.voxel_size:
        try:
            voxel_size = [float(x) for x in args.voxel_size.split(',')]
        except ValueError as e:
            print(f"Error parsing voxel size: {e}", file=sys.stderr)
            return 1
    
    try:
        view_zarr_with_neuroglancer(
            args.zarr_path, 
            args.array, 
            voxel_size, 
            args.bind_address, 
            args.bind_port,
            not args.no_browser
        )
        
        # Keep the server running until interrupted
        try:
            import time
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\nShutting down...")
        
        return 0
        
    except Exception as e:
        print(f"Error starting neuroglancer viewer: {e}", file=sys.stderr)
        return 1

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    add_arguments(parser)
    args = parser.parse_args()
    sys.exit(main(args))
"""
Command-line interface for fixing segmentation data.

This module provides commands for fixing segmentation issues like merged segments,
small objects, and holes.
"""

import argparse
import logging
import sys
import os
import numpy as np
import zarr
from typing import Optional, Dict, Any, List, Tuple

logger = logging.getLogger(__name__)


def add_arguments(parser: argparse.ArgumentParser) -> None:
    """
    Add arguments to the parser.
    
    Args:
        parser: ArgumentParser instance
    """
    parser.add_argument("input", help="Input zarr dataset path")
    parser.add_argument("output", help="Output zarr dataset path")
    parser.add_argument("--method", choices=["watershed", "erosion", "watershed-elongation"], 
                      default="watershed-elongation", 
                      help="Method to fix merged segments")
    parser.add_argument("--min-size", type=int, default=100, 
                      help="Minimum object size to keep (voxels)")
    parser.add_argument("--connectivity", type=int, choices=[6, 18, 26], default=26,
                      help="Connectivity for 3D structures (6, 18, or 26)")
    parser.add_argument("--h-min", type=int, default=3,
                      help="Minimum height for h-maxima detection (watershed methods)")
    parser.add_argument("--smoothing", type=int, default=1,
                      help="Smoothing factor for distance map (watershed methods)")
    parser.add_argument("--erosion-iterations", type=int, default=1,
                      help="Number of erosion iterations (erosion method)")
    parser.add_argument("--remove-small", action="store_true",
                      help="Remove small objects after fixing merged segments")
    parser.add_argument("--fill-holes", action="store_true",
                      help="Fill holes after fixing merged segments")
    parser.add_argument("--background-label", type=int, default=0,
                      help="Label value for background")
    parser.add_argument("--chunk-size", 
                      help="Chunk size for output zarr (comma-separated, e.g., '64,64,64')")


def parse_chunk_size(chunk_size_str: Optional[str]) -> Optional[Tuple[int, ...]]:
    """
    Parse chunk size string to tuple.
    
    Args:
        chunk_size_str: Chunk size as comma-separated values
        
    Returns:
        Tuple of integers or None if input is None
    """
    if chunk_size_str is None:
        return None
        
    try:
        return tuple(int(x) for x in chunk_size_str.split(','))
    except ValueError:
        raise ValueError(f"Invalid chunk size: {chunk_size_str}. Use format: '64,64,64'")


def main(args: argparse.Namespace) -> int:
    """
    Main entry point for fix_segmentation command.
    
    Args:
        args: Command-line arguments
        
    Returns:
        Exit code
    """
    try:
        # Import fix functions
        from segprocess.core.fix import (
            fix_merged_segments_3d_with_watershed,
            fix_merged_segments_3d_with_watershed_elongation,
            fix_merged_segments_3d_with_erosion,
            remove_small_objects_3d,
            fill_holes_3d
        )
        
        # Check if input exists
        if not os.path.exists(args.input):
            logger.error(f"Input zarr dataset not found: {args.input}")
            return 1
            
        # Open input zarr
        input_zarr = zarr.open(args.input, mode='r')
        
        # Parse chunk size
        chunk_size = parse_chunk_size(args.chunk_size)
        if chunk_size is None:
            chunk_size = input_zarr.chunks
            
        # Load segmentation data
        logger.info(f"Loading segmentation data from {args.input}")
        segmentation = input_zarr[:]
        
        # Fix merged segments based on selected method
        logger.info(f"Fixing merged segments using {args.method} method")
        if args.method == "watershed":
            fixed_segmentation, stats = fix_merged_segments_3d_with_watershed(
                segmentation,
                h_min=args.h_min,
                min_size=args.min_size,
                smoothing=args.smoothing
            )
        elif args.method == "watershed-elongation":
            fixed_segmentation, stats = fix_merged_segments_3d_with_watershed_elongation(
                segmentation,
                connectivity=args.connectivity,
                min_object_size=args.min_size,
                validation_metrics=True,
                h_min=args.h_min,
                smoothing=args.smoothing
            )
        else:  # erosion
            fixed_segmentation, stats = fix_merged_segments_3d_with_erosion(
                segmentation,
                erosion_iterations=args.erosion_iterations,
                connectivity=args.connectivity,
                min_object_size=args.min_size
            )
            
        logger.info(f"Merged segments fixed: {stats['objects_added']} objects added")
        
        # Remove small objects if requested
        if args.remove_small:
            logger.info(f"Removing small objects (min size: {args.min_size})")
            fixed_segmentation, remove_stats = remove_small_objects_3d(
                fixed_segmentation,
                min_size=args.min_size,
                background_label=args.background_label
            )
            logger.info(f"Small objects removed: {remove_stats['objects_removed']}")
            
        # Fill holes if requested
        if args.fill_holes:
            logger.info("Filling holes")
            fixed_segmentation, fill_stats = fill_holes_3d(fixed_segmentation)
            logger.info(f"Holes filled in {fill_stats['objects_with_holes']} objects")
            
        # Save output
        logger.info(f"Saving fixed segmentation to {args.output}")
        output_zarr = zarr.open(
            args.output,
            mode='w',
            shape=fixed_segmentation.shape,
            chunks=chunk_size,
            dtype=fixed_segmentation.dtype
        )
        output_zarr[:] = fixed_segmentation
        
        logger.info("Fixed segmentation saved successfully")
        return 0
        
    except Exception as e:
        logger.error(f"Error fixing segmentation: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    add_arguments(parser)
    args = parser.parse_args()
    sys.exit(main(args))
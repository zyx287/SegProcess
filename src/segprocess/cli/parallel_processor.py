"""
Command-line interface for parallel processing of segmentation data.
"""

import argparse
import pickle
import logging
import sys
import os
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

import numpy as np
import zarr

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def add_arguments(parser: argparse.ArgumentParser) -> None:
    """
    Add arguments to the parser.
    
    Args:
        parser: ArgumentParser instance
    """
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Convert command
    convert_parser = subparsers.add_parser("convert", help="Convert uint16 to uint8")
    convert_parser.add_argument("input", help="Input zarr path")
    convert_parser.add_argument("output", help="Output zarr path")
    convert_parser.add_argument("--min-val", type=int, default=34939, help="Minimum value for normalization")
    convert_parser.add_argument("--max-val", type=int, default=36096, help="Maximum value for normalization")
    convert_parser.add_argument("--chunk-size", help="Chunk size (comma-separated, e.g., '512,512,512')")
    convert_parser.add_argument("--workers", type=int, help="Number of worker processes")
    
    # Process command
    process_parser = subparsers.add_parser("process", help="Process segmentation with lookup table")
    process_parser.add_argument("input", help="Input zarr path")
    process_parser.add_argument("output", help="Output zarr path")
    process_parser.add_argument("filtered", help="Filtered output zarr path")
    process_parser.add_argument("lookup", help="Lookup table pickle file")
    process_parser.add_argument("labels", help="Valid labels pickle file")
    process_parser.add_argument("--chunk-size", help="Chunk size (comma-separated, e.g., '512,512,512')")
    process_parser.add_argument("--workers", type=int, help="Number of worker processes")
    
    # Calculate bounding boxes command
    bbox_parser = subparsers.add_parser("bbox", help="Calculate bounding boxes for cells")
    bbox_parser.add_argument("input", help="Input zarr path")
    bbox_parser.add_argument("output", help="Output pickle file")
    bbox_parser.add_argument("--labels", help="Optional labels pickle file")
    bbox_parser.add_argument("--chunk-size", help="Chunk size (comma-separated, e.g., '512,512,512')")
    bbox_parser.add_argument("--workers", type=int, help="Number of worker processes")


def parse_chunk_size(chunk_size_str: Optional[str]) -> Optional[Tuple[int, ...]]:
    """
    Parse a chunk size string into a tuple.
    
    Args:
        chunk_size_str: Chunk size as a comma-separated string
        
    Returns:
        Tuple of chunk sizes or None if chunk_size_str is None
    """
    if chunk_size_str is None:
        return None
        
    try:
        return tuple(int(x) for x in chunk_size_str.split(','))
    except ValueError:
        raise ValueError(f"Invalid chunk size: {chunk_size_str}. Use comma-separated integers, e.g., '512,512,512'")


def command_convert(args: argparse.Namespace) -> int:
    """
    Run the convert command.
    
    Args:
        args: Command-line arguments
        
    Returns:
        Exit code
    """
    try:
        from segprocess.io.converter import convert_zarr_uint16_to_uint8
        
        chunk_size = parse_chunk_size(args.chunk_size)
        
        result = convert_zarr_uint16_to_uint8(
            input_zarr_path=args.input,
            output_zarr_path=args.output,
            min_val=args.min_val,
            max_val=args.max_val,
            processing_chunk_size=chunk_size,
            num_workers=args.workers
        )
        
        logger.info(f"Conversion completed: {result}")
        return 0
        
    except Exception as e:
        logger.error(f"Error converting zarr: {e}")
        return 1


def command_process(args: argparse.Namespace) -> int:
    """
    Run the process command.
    
    Args:
        args: Command-line arguments
        
    Returns:
        Exit code
    """
    try:
        from segprocess.processing.parallel import process_zarr_dynamically
        
        # Load lookup table
        with open(args.lookup, 'rb') as f:
            lookup_table = pickle.load(f)
            
        # Load valid labels
        with open(args.labels, 'rb') as f:
            valid_labels_array = pickle.load(f)
            
        chunk_size = parse_chunk_size(args.chunk_size)
        
        # Process zarr
        process_zarr_dynamically(
            input_zarr_path=args.input,
            output_zarr_path=args.output,
            filtered_zarr_path=args.filtered,
            lookup=lookup_table,
            valid_labels_array=valid_labels_array,
            processing_chunk_size=chunk_size,
            num_workers=args.workers
        )
        
        logger.info("Processing completed")
        return 0
        
    except Exception as e:
        logger.error(f"Error processing zarr: {e}")
        return 1


def command_bbox(args: argparse.Namespace) -> int:
    """
    Run the bbox command.
    
    Args:
        args: Command-line arguments
        
    Returns:
        Exit code
    """
    try:
        from segprocess.processing.parallel import calculate_bounding_boxes_and_connectivity
        
        chunk_size = parse_chunk_size(args.chunk_size)
        
        # Load labels if provided
        cell_labels = None
        if args.labels and os.path.exists(args.labels):
            with open(args.labels, 'rb') as f:
                cell_labels = pickle.load(f)
        
        # Calculate bounding boxes
        results = calculate_bounding_boxes_and_connectivity(
            zarr_path=args.input,
            output_path=args.output,
            cell_labels=cell_labels,
            num_workers=args.workers
        )
        
        logger.info(f"Bounding box calculation completed: {len(results)} cells processed")
        return 0
        
    except Exception as e:
        logger.error(f"Error calculating bounding boxes: {e}")
        return 1


def main(args: argparse.Namespace) -> int:
    """
    Main entry point for the parallel processor.
    
    Args:
        args: Command-line arguments
        
    Returns:
        Exit code
    """
    if args.command == "convert":
        return command_convert(args)
    elif args.command == "process":
        return command_process(args)
    elif args.command == "bbox":
        return command_bbox(args)
    else:
        logger.error(f"Unknown command: {args.command}")
        return 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    add_arguments(parser)
    args = parser.parse_args()
    sys.exit(main(args))
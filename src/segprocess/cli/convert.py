"""
Command-line interface for data conversion operations.

This module provides commands for converting between different data formats
and representations, such as uint16 to uint8, zarr to tiff, etc.
"""

import argparse
import logging
import sys
import os
from typing import Optional, Tuple, Dict, Any

logger = logging.getLogger(__name__)


def add_arguments(parser: argparse.ArgumentParser) -> None:
    """
    Add arguments to the parser.
    
    Args:
        parser: ArgumentParser instance
    """
    subparsers = parser.add_subparsers(dest="subcommand", help="Conversion type")
    
    # uint16 to uint8 conversion
    uint16_parser = subparsers.add_parser("uint16-to-uint8", 
                                        help="Convert uint16 zarr to uint8 with normalization")
    uint16_parser.add_argument("input", help="Input zarr dataset path (uint16)")
    uint16_parser.add_argument("output", help="Output zarr dataset path (uint8)")
    uint16_parser.add_argument("--min-val", type=int, default=34939, 
                            help="Minimum value for normalization")
    uint16_parser.add_argument("--max-val", type=int, default=36096, 
                            help="Maximum value for normalization")
    uint16_parser.add_argument("--chunk-size", 
                            help="Processing chunk size (comma-separated, e.g., '512,512,512')")
    uint16_parser.add_argument("--storage-chunk-size", 
                            help="Storage chunk size (comma-separated, e.g., '256,256,256')")
    uint16_parser.add_argument("--num-workers", type=int, 
                            help="Number of worker processes")
    
    # Resample zarr
    resample_parser = subparsers.add_parser("resample", 
                                         help="Resample zarr dataset (downsample/upsample)")
    resample_parser.add_argument("input", help="Input zarr dataset path")
    resample_parser.add_argument("output", help="Output zarr dataset path")
    resample_parser.add_argument("factors", 
                              help="Resampling factors (comma-separated, e.g., '2,2,2')")
    resample_parser.add_argument("--method", choices=["nearest", "linear", "cubic"], 
                              default="nearest", help="Resampling method")
    resample_parser.add_argument("--chunk-size", 
                              help="Processing chunk size (comma-separated, e.g., '512,512,512')")
    resample_parser.add_argument("--storage-chunk-size", 
                              help="Storage chunk size (comma-separated, e.g., '256,256,256')")
    resample_parser.add_argument("--num-workers", type=int, 
                              help="Number of worker processes")
    
    # Zarr to precomputed
    precomputed_parser = subparsers.add_parser("to-precomputed", 
                                            help="Convert zarr to neuroglancer precomputed format")
    precomputed_parser.add_argument("input", help="Input zarr dataset path")
    precomputed_parser.add_argument("output", help="Output directory for precomputed data")
    precomputed_parser.add_argument("--format", choices=["raw", "compressed_segmentation"], 
                                 default="raw", help="Output format")
    precomputed_parser.add_argument("--resolution", default="16,16,16", 
                                 help="Resolution in nanometers (comma-separated, e.g., '16,16,16')")
    precomputed_parser.add_argument("--voxel-offset", default="0,0,0", 
                                 help="Voxel offset (comma-separated, e.g., '0,0,0')")
    precomputed_parser.add_argument("--num-workers", type=int, 
                                 help="Number of worker processes")
    
    # NPY to zarr
    npy_parser = subparsers.add_parser("npy-to-zarr", 
                                    help="Convert npy file to zarr dataset")
    npy_parser.add_argument("input", help="Input npy file path")
    npy_parser.add_argument("output", help="Output zarr dataset path")
    npy_parser.add_argument("--chunk-size", 
                         help="Chunk size (comma-separated, e.g., '64,64,64')")
    
    # TIFF to zarr
    tiff_parser = subparsers.add_parser("tiff-to-zarr", 
                                      help="Convert TIFF file(s) to zarr dataset")
    tiff_parser.add_argument("input", help="Input TIFF file path")
    tiff_parser.add_argument("output", help="Output zarr dataset path")
    tiff_parser.add_argument("--chunk-size", 
                          help="Chunk size (comma-separated, e.g., '64,64,64')")


def parse_tuple(tuple_str: str, dtype=int) -> Tuple:
    """
    Parse a string of comma-separated values to a tuple.
    
    Args:
        tuple_str: String of comma-separated values
        dtype: Data type for the values (default: int)
        
    Returns:
        Tuple of values with the specified data type
    """
    return tuple(dtype(x) for x in tuple_str.split(','))


def main(args: argparse.Namespace) -> int:
    """
    Main entry point for conversion commands.
    
    Args:
        args: Command-line arguments
        
    Returns:
        Exit code
    """
    if not hasattr(args, "subcommand") or args.subcommand is None:
        logger.error("No subcommand specified")
        return 1
    
    try:
        # Handle uint16 to uint8 conversion
        if args.subcommand == "uint16-to-uint8":
            from segprocess.io.converter import convert_zarr_uint16_to_uint8
            
            # Parse chunk sizes
            processing_chunk_size = None
            if args.chunk_size:
                processing_chunk_size = parse_tuple(args.chunk_size)
                
            storage_chunk_size = None
            if args.storage_chunk_size:
                storage_chunk_size = parse_tuple(args.storage_chunk_size)
            
            # Convert
            result = convert_zarr_uint16_to_uint8(
                input_zarr_path=args.input,
                output_zarr_path=args.output,
                min_val=args.min_val,
                max_val=args.max_val,
                storage_chunk_size=storage_chunk_size,
                processing_chunk_size=processing_chunk_size,
                num_workers=args.num_workers
            )
            
            logger.info(f"Conversion completed: {result}")
            
        # Handle resampling
        elif args.subcommand == "resample":
            from segprocess.io.converter import resample_zarr_array
            
            # Parse factors
            factors = parse_tuple(args.factors)
            
            # Parse chunk sizes
            processing_chunk_size = None
            if args.chunk_size:
                processing_chunk_size = parse_tuple(args.chunk_size)
                
            storage_chunk_size = None
            if args.storage_chunk_size:
                storage_chunk_size = parse_tuple(args.storage_chunk_size)
            
            # Resample
            result = resample_zarr_array(
                input_zarr_path=args.input,
                output_zarr_path=args.output,
                downsample_factors=factors,
                method=args.method,
                storage_chunk_size=storage_chunk_size,
                processing_chunk_size=processing_chunk_size,
                num_workers=args.num_workers
            )
            
            logger.info(f"Resampling completed: {result}")
            
        # Handle zarr to precomputed
        elif args.subcommand == "to-precomputed":
            from segprocess.io.converter import zarr_to_precomputed
            
            # Parse resolution and offset
            resolution = parse_tuple(args.resolution, float)
            voxel_offset = parse_tuple(args.voxel_offset)
            
            # Convert
            result = zarr_to_precomputed(
                zarr_path=args.input,
                output_dir=args.output,
                output_format=args.format,
                resolution=resolution,
                voxel_offset=voxel_offset,
                num_workers=args.num_workers
            )
            
            logger.info(f"Conversion to precomputed completed: {result}")
            
        # Handle npy to zarr
        elif args.subcommand == "npy-to-zarr":
            from segprocess.io.converter import npy_to_zarr
            
            # Parse chunk size
            chunk_size = None
            if args.chunk_size:
                chunk_size = parse_tuple(args.chunk_size)
            
            # Convert
            output_path = npy_to_zarr(
                input_npy_path=args.input,
                output_zarr_path=args.output,
                chunk_size=chunk_size
            )
            
            logger.info(f"NPY to zarr conversion completed: {output_path}")
            
        # Handle tiff to zarr
        elif args.subcommand == "tiff-to-zarr":
            from segprocess.io.converter import tiff_to_zarr
            
            # Parse chunk size
            chunk_size = None
            if args.chunk_size:
                chunk_size = parse_tuple(args.chunk_size)
            
            # Convert
            output_path = tiff_to_zarr(
                input_tiff_path=args.input,
                output_zarr_path=args.output,
                chunk_size=chunk_size
            )
            
            logger.info(f"TIFF to zarr conversion completed: {output_path}")
            
        else:
            logger.error(f"Unknown subcommand: {args.subcommand}")
            return 1
            
        return 0
        
    except Exception as e:
        logger.error(f"Error during conversion: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    add_arguments(parser)
    args = parser.parse_args()
    sys.exit(main(args))
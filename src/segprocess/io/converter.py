"""
Conversion utilities for segmentation data.

This module provides functions for converting between different data formats and
representations, with a focus on efficient processing of large volumes.
"""

import os
import time
import logging
import numpy as np
import zarr
from typing import Dict, List, Tuple, Optional, Union, Any, Callable
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def normalize_and_convert_to_uint8(img_uint16: np.ndarray, 
                                 min_val: int = 34939, 
                                 max_val: int = 36096) -> np.ndarray:
    """
    Normalize a uint16 image within a specific range and convert to uint8.
    
    Args:
        img_uint16: Input uint16 image
        min_val: Minimum value for normalization (default: 34939)
        max_val: Maximum value for normalization (default: 36096)
        
    Returns:
        uint8 image normalized within the specified range
    """
    # Clip the image values to the specified range
    img_clipped = np.clip(img_uint16, min_val, max_val)
    
    # Normalize to 0-1 range based on the specified min/max values
    img_normalized = (img_clipped - min_val) / (max_val - min_val)
    
    # Convert to uint8 (0-255)
    img_uint8 = (img_normalized * 255).astype(np.uint8)
    
    return img_uint8


def compute_processing_slices(total_shape: Tuple[int, ...], 
                           chunk_shape: Tuple[int, ...]) -> List[Tuple[Tuple[int, ...], List[slice]]]:
    """
    Compute slices for processing chunks in a volume.

    Args:
        total_shape: Total shape of the data (z, y, x)
        chunk_shape: Shape of each processing chunk (z, y, x)
        
    Returns:
        List of tuples, each containing:
            - Chunk indices (z_idx, y_idx, x_idx)
            - List of slices for each dimension
    """
    slices_list = []
    
    # Calculate number of chunks in each dimension
    chunks_per_dim = [int(np.ceil(total_shape[dim] / chunk_shape[dim])) 
                     for dim in range(len(total_shape))]
    
    for chunk_indices in np.ndindex(*chunks_per_dim):
        # Calculate starting offset for this chunk
        volume_offset = [idx * chunk_shape[dim] for dim, idx in enumerate(chunk_indices)]
        
        # Calculate actual chunk size (handling boundary cases)
        volume_size = [
            min(chunk_shape[dim], total_shape[dim] - volume_offset[dim])
            for dim in range(len(chunk_shape))
        ]
        
        # Create slices for each dimension
        slices = [
            slice(volume_offset[dim], volume_offset[dim] + volume_size[dim])
            for dim in range(len(volume_offset))
        ]
        
        slices_list.append((chunk_indices, slices))
    
    return slices_list


def is_chunk_processed(output_zarr: zarr.core.Array, slices: List[slice]) -> bool:
    """
    Check if a chunk has already been processed in the output Zarr array.
    
    Args:
        output_zarr: Output Zarr array
        slices: List of slices defining the chunk
        
    Returns:
        True if the chunk has been processed, False otherwise
    """
    try:
        # Create a small sample from the chunk to check if it contains data
        sample_slices = []
        for s in slices:
            start = s.start
            end = min(s.start + 10, s.stop)  # Sample first 10 elements or fewer
            sample_slices.append(slice(start, end))
        
        sample = output_zarr[tuple(sample_slices)]
        return np.any(sample)
    except Exception as e:
        logger.debug(f"Error checking if chunk is processed: {e}")
        return False


def convert_processing_chunk(chunk_info: Tuple[Tuple[int, ...], List[slice]],
                          input_zarr_path: str,
                          output_zarr_path: str,
                          min_val: int,
                          max_val: int) -> None:
    """
    Convert a processing chunk from uint16 to uint8 with normalization.

    Args:
        chunk_info: Tuple containing chunk indices and slices
        input_zarr_path: Path to the input Zarr array
        output_zarr_path: Path to the output Zarr array
        min_val: Minimum value for normalization
        max_val: Maximum value for normalization
    """
    chunk_indices, slices = chunk_info
    
    try:
        # Open zarr arrays
        input_zarr = zarr.open(input_zarr_path, mode='r')
        output_zarr = zarr.open(output_zarr_path, mode='a')
        
        # Check if already processed
        if is_chunk_processed(output_zarr, slices):
            logger.info(f"Skipping already processed chunk: {chunk_indices}")
            return
        
        # Load chunk
        start_time = time.time()
        chunk_data = input_zarr[tuple(slices)]
        load_time = time.time() - start_time
        
        # Convert chunk
        start_time = time.time()
        converted_chunk = normalize_and_convert_to_uint8(chunk_data, min_val, max_val)
        convert_time = time.time() - start_time
        
        # Write converted chunk
        start_time = time.time()
        output_zarr[tuple(slices)] = converted_chunk
        write_time = time.time() - start_time
        
        total_time = load_time + convert_time + write_time
        chunk_shape = tuple(s.stop - s.start for s in slices)
        chunk_size_gb = np.prod(chunk_shape) * 2 / (1024**3)  # uint16 input size in GB
        
        logger.info(f"Processed chunk {chunk_indices}: shape={chunk_shape}, "
                  f"size={chunk_size_gb:.2f}GB, time={total_time:.2f}s "
                  f"(load={load_time:.2f}s, convert={convert_time:.2f}s, write={write_time:.2f}s)")
    except Exception as e:
        logger.error(f"Error processing chunk {chunk_indices}: {e}")
        raise


def convert_zarr_uint16_to_uint8(input_zarr_path: str,
                              output_zarr_path: str,
                              min_val: int = 34939,
                              max_val: int = 36096,
                              storage_chunk_size: Optional[Tuple[int, ...]] = None,
                              processing_chunk_size: Tuple[int, ...] = (1024, 1024, 1024),
                              num_workers: Optional[int] = None) -> Dict[str, Any]:
    """
    Convert a uint16 Zarr array to uint8 using multiprocessing.
    
    Args:
        input_zarr_path: Path to the input Zarr array (uint16)
        output_zarr_path: Path to the output Zarr array (uint8)
        min_val: Minimum value for normalization (default: 34939)
        max_val: Maximum value for normalization (default: 36096)
        storage_chunk_size: Physical chunk size for Zarr storage (default: None, use input chunks)
        processing_chunk_size: Size of data each worker processes (default: 1024x1024x1024)
        num_workers: Number of worker processes (default: None, use CPU count)
        
    Returns:
        Dictionary with processing statistics
    """
    from segprocess.processing.multiprocess import process_large_zarr
    
    @process_large_zarr
    def _normalize_convert(data: np.ndarray, min_val: int = 34939, max_val: int = 36096) -> np.ndarray:
        """Normalize and convert uint16 data to uint8"""
        return normalize_and_convert_to_uint8(data, min_val, max_val)
    
    # Run the conversion using the multiprocessing framework
    result = _normalize_convert(
        input_zarr_path=input_zarr_path,
        output_zarr_path=output_zarr_path,
        output_dtype=np.uint8,
        storage_chunk_size=storage_chunk_size,
        processing_chunk_size=processing_chunk_size,
        num_workers=num_workers,
        min_val=min_val,
        max_val=max_val
    )
    
    return result


def resample_zarr_array(input_zarr_path: str,
                     output_zarr_path: str,
                     downsample_factors: Tuple[int, ...],
                     method: str = 'nearest',
                     storage_chunk_size: Optional[Tuple[int, ...]] = None,
                     processing_chunk_size: Optional[Tuple[int, ...]] = None,
                     num_workers: Optional[int] = None) -> Dict[str, Any]:
    """
    Resample a Zarr array with specified downsampling factors.
    
    Args:
        input_zarr_path: Path to the input Zarr array
        output_zarr_path: Path to the output Zarr array
        downsample_factors: Downsampling factors for each dimension
        method: Resampling method ('nearest', 'linear', 'cubic')
        storage_chunk_size: Physical chunk size for Zarr storage
        processing_chunk_size: Size of data each worker processes
        num_workers: Number of worker processes
        
    Returns:
        Dictionary with processing statistics
    """
    from segprocess.processing.multiprocess import process_large_zarr
    from scipy.ndimage import zoom
    
    @process_large_zarr
    def _resample(data: np.ndarray, 
                downsample_factors: Tuple[int, ...], 
                method: str = 'nearest') -> np.ndarray:
        """Resample data with specified downsampling factors"""
        # Convert downsampling factors to zoom factors (reciprocal)
        zoom_factors = tuple(1.0 / factor for factor in downsample_factors)
        
        # Map method string to order parameter for scipy.ndimage.zoom
        order_map = {
            'nearest': 0,
            'linear': 1,
            'cubic': 3
        }
        order = order_map.get(method, 0)
        
        # Resample the data
        resampled = zoom(data, zoom_factors, order=order)
        return resampled
    
    # Open input zarr to get shape and dtype
    input_zarr = zarr.open(input_zarr_path, mode='r')
    input_shape = input_zarr.shape
    
    # Calculate output shape
    output_shape = tuple(int(s // f) for s, f in zip(input_shape, downsample_factors))
    
    # Set default processing chunk size if not specified
    if processing_chunk_size is None:
        # Choose a reasonable processing chunk size (smaller than input chunks)
        processing_chunk_size = tuple(min(c, 512) for c in input_zarr.chunks)
    
    # Set default storage chunk size if not specified
    if storage_chunk_size is None:
        # Choose a reasonable storage chunk size (similar to input chunks but smaller)
        storage_chunk_size = tuple(min(c // f, 256) for c, f in zip(input_zarr.chunks, downsample_factors))
    
    # Run the resampling using the multiprocessing framework
    result = _resample(
        input_zarr_path=input_zarr_path,
        output_zarr_path=output_zarr_path,
        output_dtype=input_zarr.dtype,
        storage_chunk_size=storage_chunk_size,
        processing_chunk_size=processing_chunk_size,
        num_workers=num_workers,
        downsample_factors=downsample_factors,
        method=method
    )
    
    return result


def zarr_to_precomputed(zarr_path: str,
                      output_dir: str,
                      output_format: str = 'raw',
                      resolution: Tuple[float, ...] = (8, 8, 8),
                      voxel_offset: Tuple[int, ...] = (0, 0, 0),
                      num_workers: Optional[int] = None) -> Dict[str, Any]:
    """
    Convert a Zarr array to Neuroglancer precomputed format.
    
    Args:
        zarr_path: Path to the input Zarr array
        output_dir: Output directory for precomputed data
        output_format: Output format ('raw' or 'compressed_segmentation')
        resolution: Resolution in nanometers (x, y, z)
        voxel_offset: Voxel offset (x, y, z)
        num_workers: Number of worker processes
        
    Returns:
        Dictionary with processing statistics
        
    Raises:
        ImportError: If cloudvolume is not available
    """
    try:
        import cloudvolume
        from cloudvolume import CloudVolume
    except ImportError:
        raise ImportError("cloudvolume is required for this function. Install it with: pip install cloud-volume")
    
    # Get info from zarr array
    zarr_array = zarr.open(zarr_path, mode='r')
    
    # Determine layer type based on data type
    if np.issubdtype(zarr_array.dtype, np.integer) and zarr_array.dtype != np.uint8:
        layer_type = 'segmentation'
    else:
        layer_type = 'image'
    
    # Create info for CloudVolume
    info = CloudVolume.create_new_info(
        num_channels=1,
        layer_type=layer_type,
        data_type=str(zarr_array.dtype),
        encoding=output_format,
        resolution=resolution,
        voxel_offset=voxel_offset,
        volume_size=zarr_array.shape,
        chunk_size=zarr_array.chunks
    )
    
    # Create CloudVolume
    vol = CloudVolume(
        f'file://{output_dir}',
        info=info,
        bounded=True,
        compress='gzip' if output_format == 'raw' else None,
        progress=True
    )
    
    # Commit info and provenance
    vol.commit_info()
    vol.provenance.processing.append({
        'method': 'zarr_to_precomputed',
        'source': zarr_path,
        'date': time.strftime('%Y-%m-%d %H:%M %Z')
    })
    vol.commit_provenance()
    
    # Define chunk processing function
    def process_chunk(chunk_info):
        chunk_indices, slices = chunk_info
        
        try:
            # Load chunk from zarr
            chunk_data = zarr_array[tuple(slices)]
            
            # Get absolute slices for CloudVolume
            abs_slices = tuple(slice(s.start, s.stop) for s in slices)
            
            # Write chunk to CloudVolume
            vol[abs_slices] = chunk_data
            
            logger.info(f"Processed chunk {chunk_indices}")
            return True
        except Exception as e:
            logger.error(f"Error processing chunk {chunk_indices}: {e}")
            return False
    
    # Process chunks
    from concurrent.futures import ThreadPoolExecutor
    
    # Compute chunks
    chunks = compute_processing_slices(zarr_array.shape, zarr_array.chunks)
    
    # Set number of workers
    if num_workers is None:
        import multiprocessing
        num_workers = multiprocessing.cpu_count()
    
    # Process chunks in parallel
    success_count = 0
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        results = list(executor.map(process_chunk, chunks))
        success_count = sum(results)
    
    logger.info(f"Conversion completed: {success_count}/{len(chunks)} chunks processed")
    
    return {
        'status': 'completed',
        'total_chunks': len(chunks),
        'successful_chunks': success_count,
        'failed_chunks': len(chunks) - success_count,
        'output_path': output_dir
    }


def npy_to_zarr(input_npy_path: str,
             output_zarr_path: str,
             chunk_size: Optional[Tuple[int, ...]] = None) -> str:
    """
    Convert a numpy array file to zarr format without loading the full array into memory.
    
    Args:
        input_npy_path: Path to the input numpy array file
        output_zarr_path: Path to the output zarr array
        chunk_size: Chunk size for the zarr array (default: auto)
        
    Returns:
        Path to the output zarr array
    """
    # Load the array metadata without loading the data
    array = np.load(input_npy_path, mmap_mode="r")
    
    # Determine a reasonable chunk size if not specified
    if chunk_size is None:
        # Default to ~64MB chunks
        target_chunk_bytes = 64 * 1024 * 1024
        bytes_per_element = array.dtype.itemsize
        
        # Calculate number of elements per chunk
        elements_per_chunk = max(1, int(target_chunk_bytes / bytes_per_element))
        
        # Distribute elements across dimensions
        chunk_size = []
        remaining_elements = elements_per_chunk
        
        for dim_size in array.shape:
            # Take the cube root of remaining elements for each dimension
            # but cap at the dimension size
            chunk_dim = min(dim_size, max(1, int(remaining_elements ** (1.0 / len(array.shape)))))
            chunk_size.append(chunk_dim)
            remaining_elements = max(1, remaining_elements // chunk_dim)
    
    # Create the zarr array
    zarr_array = zarr.open(
        output_zarr_path,
        mode="w",
        shape=array.shape,
        chunks=chunk_size,
        dtype=array.dtype
    )
    
    # Process the data in chunks to avoid loading the whole array
    for index in np.ndindex(*(len(array.shape) * [2])):
        # Calculate slices for this chunk (half of each dimension)
        slices = []
        for i, idx in enumerate(index):
            dim_size = array.shape[i]
            half_size = dim_size // 2
            slices.append(slice(idx * half_size, min((idx + 1) * half_size, dim_size)))
        
        # Load and write this chunk
        chunk_data = array[tuple(slices)]
        zarr_array[tuple(slices)] = chunk_data
        
        logger.info(f"Processed chunk at {slices}")
    
    logger.info(f"Converted {input_npy_path} to {output_zarr_path}")
    logger.info(f"Shape: {array.shape}, Dtype: {array.dtype}, Chunks: {chunk_size}")
    
    return output_zarr_path


def tiff_to_zarr(input_tiff_path: str,
              output_zarr_path: str,
              chunk_size: Optional[Tuple[int, ...]] = None) -> str:
    """
    Convert a TIFF file to zarr format without loading the full array into memory.
    
    Args:
        input_tiff_path: Path to the input TIFF file
        output_zarr_path: Path to the output zarr array
        chunk_size: Chunk size for the zarr array (default: auto)
        
    Returns:
        Path to the output zarr array
        
    Raises:
        ImportError: If tifffile is not available
    """
    try:
        import tifffile
    except ImportError:
        raise ImportError("tifffile is required for this function. Install it with: pip install tifffile")
    
    # Open the TIFF file
    with tifffile.TiffFile(input_tiff_path) as tif:
        # Get the first page to determine dtype and shape
        page = tif.pages[0]
        dtype = page.dtype
        
        if tif.series:
            # Get the shape from the series (handles multi-page TIFFs properly)
            shape = tif.series[0].shape
        else:
            # Fallback: assume it's a stack of 2D images
            shape = (len(tif.pages),) + page.shape
        
        # Determine a reasonable chunk size if not specified
        if chunk_size is None:
            # For 3D data, use ~64MB chunks
            target_chunk_bytes = 64 * 1024 * 1024
            bytes_per_element = dtype.itemsize
            
            # Aim for cubic chunks where possible
            elements_per_chunk = max(1, int(target_chunk_bytes / bytes_per_element))
            chunk_dim = max(1, int(elements_per_chunk ** (1.0 / len(shape))))
            
            # Create the chunk size tuple
            chunk_size = tuple(min(dim_size, chunk_dim) for dim_size in shape)
        
        # Create the zarr array
        zarr_array = zarr.open(
            output_zarr_path,
            mode="w",
            shape=shape,
            chunks=chunk_size,
            dtype=dtype
        )
        
        # For 3D data, process by z-slices
        if len(shape) == 3:
            for z in range(shape[0]):
                # Read a single z-slice
                data = tif.asarray(key=z)
                
                # Write to zarr
                zarr_array[z:z+1, :, :] = data.reshape(1, *data.shape)
                
                if z % 100 == 0:
                    logger.info(f"Processed slice {z}/{shape[0]}")
        
        # For other data, try to read and write directly
        else:
            data = tif.asarray()
            zarr_array[:] = data
    
    logger.info(f"Converted {input_tiff_path} to {output_zarr_path}")
    logger.info(f"Shape: {shape}, Dtype: {dtype}, Chunks: {chunk_size}")
    
    return output_zarr_path


# Make functions available at the module level
__all__ = [
    'convert_zarr_uint16_to_uint8',
    'normalize_and_convert_to_uint8',
    'resample_zarr_array',
    'zarr_to_precomputed',
    'npy_to_zarr',
    'tiff_to_zarr',
    'compute_processing_slices',
    'is_chunk_processed'
]
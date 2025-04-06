"""
Utility functions for multiprocessing operations.
"""

import numpy as np
import logging
from typing import List, Tuple, Dict, Any, Union

logger = logging.getLogger(__name__)

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


def zarr_chunk_exists(zarr_array, slices: List[slice]) -> bool:
    """
    Check if a chunk exists and has data in a Zarr array.
    
    Args:
        zarr_array: Zarr array
        slices: List of slices defining the chunk
        
    Returns:
        True if the chunk exists and has data, False otherwise
    """
    try:
        # Create a small sample from the chunk to check if it contains data
        sample_slices = []
        for s in slices:
            start = s.start
            end = min(s.start + 10, s.stop)  # Sample first 10 elements or fewer
            sample_slices.append(slice(start, end))
        
        sample = zarr_array[tuple(sample_slices)]
        return np.any(sample)
    except Exception as e:
        logger.debug(f"Error checking if chunk exists: {e}")
        return False


def estimate_memory_usage(shape: Tuple[int, ...], 
                         dtype: np.dtype, 
                         factor: float = 3.0) -> int:
    """
    Estimate memory usage for processing a chunk of data.
    
    Args:
        shape: Shape of the data
        dtype: Data type
        factor: Memory usage factor (default: 3.0, assuming input, output, and temporary data)
        
    Returns:
        Estimated memory usage in bytes
    """
    element_size = np.dtype(dtype).itemsize
    num_elements = np.prod(shape)
    return int(num_elements * element_size * factor)


def generate_batch_tasks(total_tasks: int, batch_size: int) -> List[Tuple[int, int]]:
    """
    Generate batches of task indices.
    
    Args:
        total_tasks: Total number of tasks
        batch_size: Size of each batch
        
    Returns:
        List of tuples containing (start_idx, end_idx) for each batch
    """
    batches = []
    for start_idx in range(0, total_tasks, batch_size):
        end_idx = min(start_idx + batch_size, total_tasks)
        batches.append((start_idx, end_idx))
    return batches


# Make functions available at the module level
__all__ = [
    'compute_processing_slices',
    'zarr_chunk_exists',
    'estimate_memory_usage',
    'generate_batch_tasks'
]
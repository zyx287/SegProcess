'''
author: zyx
date: 2025-03-10
last modified: 2025-03-10
description: 
    Frequently used functions for multiprocessing
'''
import numpy as np


def compute_processing_slices(total_volume_size, processing_chunk_size):
    '''
    Precompute slices for processing chunks in the volume.

    Parms:
        total_volume_size: tuple
            The total size of the volume (x, y, z).
        processing_chunk_size: tuple
            The size of each processing chunk (x, y, z).
    
    Returns:
        slices_list: list
            List of tuples, each: (chunk_indices, slices)
            chunk_indices: tuple
                Indices of the current chunk in the total volume.
                e.g. (0, 0, 0)
            slices: list of slice objects
                Slices for the current chunk in each dimension.
                e.g. slice(0, 1024, None), slice(0, 1024, None), slice(0, 1024, None)
    Usage:
        slices_list = compute_processing_slices(total_volume_size, processing_chunk_size)
    '''
    slices_list = []
    
    # Calculate number of chunks in each dimension
    chunks_per_dim = [int(np.ceil(total_volume_size[dim] / processing_chunk_size[dim])) 
                     for dim in range(len(total_volume_size))]
    
    for chunk_indices in np.ndindex(*chunks_per_dim):
        # Calculate starting offset for this chunk
        volume_offset = [idx * processing_chunk_size[dim] for dim, idx in enumerate(chunk_indices)]
        
        # Calculate actual chunk size (handling boundary cases)
        volume_size = [
            min(processing_chunk_size[dim], total_volume_size[dim] - volume_offset[dim])
            for dim in range(len(processing_chunk_size))
        ]
        
        slices = [
            slice(volume_offset[dim], volume_offset[dim] + volume_size[dim])
            for dim in range(len(volume_offset))
        ]

        slices_list.append((chunk_indices, slices))
    
    return slices_list
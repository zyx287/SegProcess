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
    
    Args:
        total_volume_size: tuple
            The total size of the volume (z, y, x).
        processing_chunk_size: tuple
            The size of each processing chunk (z, y, x).
            
    Returns:
        list of tuple
            A list of slices for each processing chunk.
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
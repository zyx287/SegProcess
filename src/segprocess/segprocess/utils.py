'''
author: zyx
date: 2024-11-24
description: 
    Functions for processing segmentation data
'''
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed

def id_to_label(data_chunk, lookup):
    '''
    Process dense labeled segmentation data using the look-up table
    Parms:
        data_chunk: np.ndarray
            The dense labeled segmentation data
        lookup: dict
            The look-up table
    '''
    print('Processing chunk...')
    vectorized_id_to_label = np.vectorize(lookup.get, otypes=[object])
    result = vectorized_id_to_label(data_chunk)
    print("Finished processing chunk.")
    return result


def process_large_volume(data_volume, lookup, num_threads=8, chunk_size=64):
    """
    Process a large numpy array volume using multi-threading.
    Parms:
        data_volume: np.ndarray
            Large volume to process.
        lookup: dict
            Look-up table for mapping.
        num_threads: int
            Number of threads for parallel processing.
        chunk_size: int
            Size of chunks along the first dimension (z-axis).
    Return:
        np.ndarray
            Processed volume.
    """
    chunks = [(i, data_volume[i:i+chunk_size]) for i in range(0, data_volume.shape[0], chunk_size)]
    results = [None] * len(chunks)

    def process_chunk(idx, chunk):
        return idx, id_to_label(chunk, lookup)
    
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = {executor.submit(process_chunk, idx, chunk): idx for idx, chunk in chunks}
        for future in as_completed(futures):
            idx, result = future.result()
            results[idx // chunk_size] = result

    processed_volume = np.concatenate(results, axis=0)
    return processed_volume
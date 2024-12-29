'''
author: zyx
date: 2024-11-24
description: 
    Functions for processing segmentation data
'''
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
import dask.array as da
from dask.distributed import Client, LocalCluster
from dask.diagnostics import ProgressBar

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

def process_volume_dask(data, lookup, chunk_size=(100, 100, 100), mode="threads", n_cpus=4):
    '''
    Process a large 3D volume using a lookup table with Dask, including chunk tracking and execution mode selection.
    Parms:
        data: np.ndarray
            The large 3D volume to process.
        lookup: dict
            A lookup table where keys map to labels.
        chunk_size: tuple
            Size of 3D chunks to divide the volume into for processing.
        mode: str
            Execution mode: "threads", "processes", or "distributed".
        n_cpus: int
            Number of CPUs or workers to use for computation.
    Returns:
        np.ndarray
            The processed 3D volume with labels applied.
    Usage:
        result, _, __, = process_large_3d_volume_with_lookup(
            data=z_array,
            lookup=data,  # Assuming `data` is the lookup dictionary
            chunk_size=(64, 64, 64),  # Adjust chunk size based on memory
            mode="threads",  # Options: "threads", "processes", "distributed"
            n_cpus=32  # Number of CPUs or workers)
    '''
    processed_chunks = []#track processed chunks
    def apply_lookup(data_chunk, block_id=None):
        '''
        id_to_label function for Dask map_blocks
        '''
        try:
            if not isinstance(block_id, tuple):
                print(f"Unexpected block_id: {block_id}. Skipping.")
                return None

            print(f"Processing chunk {block_id} with shape {data_chunk.shape}")
            vectorized_lookup = np.vectorize(lookup.get, otypes=[object])
            result = vectorized_lookup(data_chunk)
            processed_chunks.append(block_id)

            return result
        except Exception as e:
            print(f"Error processing chunk {block_id}: {e}")
            raise
    dask_array = da.from_array(data, chunks=chunk_size)

    processed_dask_array = dask_array.map_blocks(apply_lookup, dtype=object, block_id=True)

    client = None
    if mode == "distributed":
        cluster = LocalCluster(n_workers=n_cpus, threads_per_worker=1)
        client = Client(cluster)
        print(f"Running on distributed cluster with {n_cpus} workers.")
    elif mode == "processes":
        scheduler = "processes"
        print(f"Running in multiprocessing mode on {n_cpus} CPUs.")
    else:# Default to threads
        scheduler = "threads"
        print(f"Running in multithreaded mode on {n_cpus} threads.")

    with ProgressBar():
        result = processed_dask_array.compute(scheduler=scheduler if mode != "distributed" else None)

    # Verify all chunks were processed
    print("Processed chunks:", processed_chunks)
    processed_chunks_number = len(processed_chunks)
    total_chunks_number = dask_array.npartitions
    # assert len(processed_chunks) == dask_array.npartitions, "Not all chunks were processed!"

    if client:
        client.close()

    print("Processing completed.")
    return result, processed_chunks_number, total_chunks_number

def process_and_save_zarr_dask(input_zarr_path, output_zarr_path, lookup, chunk_size=(64, 64, 64), mode="threads", n_cpus=4):
    '''
    Read, process, and save a large Zarr dataset using Dask, without loading the full dataset into memory.

    Parameters:
        input_zarr_path: str
            Path to the input Zarr dataset.
        output_zarr_path: str
            Path to save the processed Zarr dataset.
        lookup: dict
            A lookup table where keys map to labels.
        chunk_size: tuple
            Size of 3D chunks for processing.
        mode: str
            Execution mode: "threads", "processes", or "distributed".
        n_cpus: int
            Number of CPUs or workers to use for computation.
    '''
    processed_chunks = []
    def apply_lookup(data_chunk, block_id=None):
        try:
            if block_id is not None:
                processed_chunks.append(block_id)

            print(f"Processing chunk {block_id} with shape {data_chunk.shape}")
            vectorized_lookup = np.vectorize(lambda x: lookup.get(x, 0), otypes=[np.uint32])
            result = vectorized_lookup(data_chunk)
            return result
        except Exception as e:
            print(f"Error processing chunk {block_id}: {e}")
            raise

    dask_array = da.from_zarr(input_zarr_path).rechunk(chunk_size)

    processed_dask_array = dask_array.map_blocks(apply_lookup, dtype=np.uint32)

    client = None
    if mode == "distributed":
        cluster = LocalCluster(n_workers=n_cpus, threads_per_worker=1)
        client = Client(cluster)
        print(f"Running on distributed cluster with {n_cpus} workers.")
    elif mode == "processes":
        scheduler = "processes"
        print(f"Running in multiprocessing mode on {n_cpus} CPUs.")
    else:  # Default to threads
        scheduler = "threads"
        print(f"Running in multithreaded mode on {n_cpus} threads.")

    with ProgressBar():
        processed_dask_array.to_zarr(output_zarr_path, overwrite=True, compute=True)

    total_chunks_number = dask_array.npartitions
    processed_chunks_number = len(processed_chunks)
    print(f"Processed chunks: {processed_chunks_number}/{total_chunks_number}")
    # assert processed_chunks_number == total_chunks_number, "Not all chunks were processed!"

    if client:
        client.close()

    print(f"Processing completed. Processed data saved to {output_zarr_path}")
    return processed_chunks_number, total_chunks_number

################ Legacy code ################
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
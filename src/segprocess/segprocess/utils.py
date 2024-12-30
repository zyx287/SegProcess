'''
author: zyx
date: 2024-11-24
description: 
    Functions for processing segmentation data
'''
import numpy as np
import zarr
from concurrent.futures import ThreadPoolExecutor, as_completed
import dask.array as da
from dask.distributed import Client, LocalCluster
from dask.diagnostics import ProgressBar
from multiprocessing import Process, JoinableQueue, cpu_count
import pickle

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

def filter_labels(data_array, labels):
    '''
    Filter out labels from a dense labeled segmentation data
    Parms:
        data_array: np.ndarray
            The dense labeled segmentation data
        labels: array
            The labels to keep
    Returns:
        np.ndarray
            The filtered dense labeled segmentation data
    '''
    data_array = np.where(data_array == None, 0, data_array)
    vaild_labels_set = set(labels.flatten())
    def _filter_labels(x):
        return x if x in vaild_labels_set else 0
    vectorized_filter_labels = np.vectorize(_filter_labels)
    return vectorized_filter_labels(data_array)

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

def npy_to_zarr(input_npy_path, output_zarr_path, chunk_size=(64, 64, 64)):
    '''
    Convert NPY to Zarr format (without loading the full dataset into memory).
    moveaxis is recommended to apply right before igneous processing
    '''
    data_array = np.load(input_npy_path, mmap_mode="r")
    zarr_array = zarr.open(output_zarr_path, mode="w", shape=data_array.shape, chunks=chunk_size, dtype=data_array.dtype)
    zarr_array[:] = data_array
    print(f"Data saved to {output_zarr_path}")

#### Multi-process pipeline ####
def process_and_filter_chunk(task, input_zarr_path, output_zarr_path, filtered_zarr_path, lookup, valid_labels_set):
    '''
    Process and filter a single chunk based on the task index.

    Parms:
        task: tuple
            Task containing the chunk index and slices.
        input_zarr_path: str
            Path to the input Zarr dataset.
        output_zarr_path: str
            Path to the processed Zarr dataset.
        filtered_zarr_path: str
            Path to the filtered Zarr dataset.
        lookup: dict
            A lookup table mapping input values to output values.
        valid_labels_set: set
            A set of valid labels for filtering.
    '''
    index, slices = task

    input_zarr = zarr.open(input_zarr_path, mode="r")
    output_zarr = zarr.open(output_zarr_path, mode="a")
    filtered_zarr = zarr.open(filtered_zarr_path, mode="a")

    data_chunk = input_zarr[tuple(slices)]

    vectorized_lookup = np.vectorize(lambda x: lookup.get(x, 0), otypes=[np.uint32])
    processed_chunk = vectorized_lookup(data_chunk)

    processed_chunk = np.where(processed_chunk == 'None', 0, processed_chunk)  # Replace 'None' with 0
    vectorized_filter = np.vectorize(lambda value: value if value in valid_labels_set else 0)
    filtered_chunk = vectorized_filter(processed_chunk)

    output_zarr[tuple(slices)] = processed_chunk
    filtered_zarr[tuple(slices)] = filtered_chunk

    print(f"Processed and filtered chunk at index {index}")

def worker(task_queue, input_zarr_path, output_zarr_path, filtered_zarr_path, lookup, valid_labels_set):
    while True:
        task = task_queue.get()
        if task is None:
            break
        process_and_filter_chunk(task, input_zarr_path, output_zarr_path, filtered_zarr_path, lookup, valid_labels_set)
        task_queue.task_done()

def process_zarr(input_zarr_path, output_zarr_path, filtered_zarr_path, lookup, valid_labels_array, num_workers=None):
    '''
    Dynamically process and filter a Zarr dataset using multiprocessing.

    Parms:
        input_zarr_path: str
            Path to the input Zarr dataset.
        output_zarr_path: str
            Path to the processed Zarr dataset.
        filtered_zarr_path: str
            Path to the filtered Zarr dataset.
        lookup: dict
            A lookup table mapping input values to output values.
        valid_labels_array: np.ndarray
            An array of valid labels for filtering.
        num_workers: int, optional
            Number of worker processes to use. Defaults to the number of CPUs.
    Usage:
        Load the lookup table and valid labels array from files:
        ```python
        with open('/home/zhangy8@hhmi.org/data1/20241226_reconstruct/output/id_to_label_20241226.pkl', 'rb') as f:
            lookup_table = pickle.load(f)
        with open('/home/zhangy8@hhmi.org/data1/20241226_reconstruct/output/proofread_labels_20241229.pkl', 'rb') as f:
            valid_labels_array = pickle.load(f)
        ```
        Process the Zarr dataset:
        ```python
        process_zarr_dynamically(
            input_zarr_path=input_zarr_path,
            output_zarr_path=output_zarr_path,
            filtered_zarr_path=filtered_zarr_path,
            lookup=lookup_table,
            valid_labels_array=valid_labels_array,
            num_workers=64  # Adjust based on available CPUs
        )
        ```
    '''
    input_zarr = zarr.open(input_zarr_path, mode="r")
    output_zarr = zarr.open(
        output_zarr_path, mode="w", shape=input_zarr.shape, chunks=input_zarr.chunks, dtype="uint32"
    )
    filtered_zarr = zarr.open(
        filtered_zarr_path, mode="w", shape=input_zarr.shape, chunks=input_zarr.chunks, dtype="uint32"
    )

    valid_labels_set = set(valid_labels_array.flatten())# Proofread labels should be calculated

    chunk_shape = input_zarr.chunks
    total_shape = input_zarr.shape
    ndim = input_zarr.ndim

    task_index = 0
    task_queue = JoinableQueue()
    for chunk_indices in np.ndindex(*[int(np.ceil(total_shape[dim] / chunk_shape[dim])) for dim in range(ndim)]):
        slices = [
            slice(idx * chunk_shape[dim], min((idx + 1) * chunk_shape[dim], total_shape[dim]))
            for dim, idx in enumerate(chunk_indices)
        ]
        task_queue.put((task_index, slices))
        task_index += 1

    num_workers = num_workers or cpu_count()
    workers = []
    for _ in range(num_workers):
        worker_process = Process(target=worker, args=(
            task_queue, input_zarr_path, output_zarr_path, filtered_zarr_path, lookup, valid_labels_set
        ))
        worker_process.start()
        workers.append(worker_process)
    task_queue.join()
    for _ in range(num_workers):
        task_queue.put(None)
    for worker_process in workers:
        worker_process.join()

    print("Processing completed.")


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
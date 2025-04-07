import zarr
import numpy as np
from multiprocessing import Process, JoinableQueue, Queue, cpu_count
import pickle
import networkx as nx

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

def process_zarr_dynamically(input_zarr_path, output_zarr_path, filtered_zarr_path, lookup, valid_labels_array, num_workers=None):
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
    '''
    input_zarr = zarr.open(input_zarr_path, mode="r")
    output_zarr = zarr.open(
        output_zarr_path, mode="w", shape=input_zarr.shape, chunks=input_zarr.chunks, dtype="uint32"
    )
    filtered_zarr = zarr.open(
        filtered_zarr_path, mode="w", shape=input_zarr.shape, chunks=input_zarr.chunks, dtype="uint32"
    )

    valid_labels_set = set(valid_labels_array.flatten())  # Proofread labels should be calculated

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

    __all__ = [
        'process_zarr_dynamically'
    ]
'''
author: zyx
date: 2025-02-10
last modified: 2025-02-10
description: 
    Step1: Generate all the mitochondria label using multi-processing
'''
import zarr
import numpy as np
import multiprocessing
from tqdm import tqdm
import os
import pickle

zarr_path = "/media/zhangy8/ca0155b9-932b-4491-8ea9-d40a586475cf/SegmentationData/source/mitochondria_seg_mag2_20250206.zarr"
output_dir = "/home/zhangy8@hhmi.org/data1/20250210_assign_mitochondria/Output"
final_output_path = "/home/zhangy8@hhmi.org/data1/20250210_assign_mitochondria/Output/final/unique_labels.pkl"
processed_chunks_path = "/home/zhangy8@hhmi.org/data1/20250210_assign_mitochondria/Output/final/processed_chunks.pkl"

os.makedirs(output_dir, exist_ok=True)
zarr_data = zarr.open(zarr_path, mode='r')

chunk_size = (256, 256, 256)
num_workers = 80


def process_chunk(chunk_idx, chunk):
    """
    Extract unique labels from a chunk of the zarr dataset.
    Saves results to a file.
    """
    unique_labels = np.unique(chunk)

    chunk_filename = os.path.join(output_dir, f"labels_chunk_{chunk_idx}.pkl")
    with open(chunk_filename, "wb") as f:
        pickle.dump(set(unique_labels), f)

    return chunk_idx


def parallel_process():
    """
    Processes the zarr dataset in parallel, extracting unique labels per chunk.
    Tracks progress to ensure all chunks are processed.
    """
    tasks = []
    pool = multiprocessing.Pool(processes=num_workers)
    processed_chunks = set()

    if os.path.exists(processed_chunks_path):
        with open(processed_chunks_path, "rb") as f:
            processed_chunks = pickle.load(f)

    chunk_idx = 0
    expected_chunks = []

    print("Extracting unique labels per chunk...")
    for x in tqdm(range(0, zarr_data.shape[0], chunk_size[0])):
        for y in range(0, zarr_data.shape[1], chunk_size[1]):
            for z in range(0, zarr_data.shape[2], chunk_size[2]):
                expected_chunks.append(chunk_idx)
                if chunk_idx in processed_chunks:
                    chunk_idx += 1
                    continue

                chunk = zarr_data[x:x+chunk_size[0], y:y+chunk_size[1], z:z+chunk_size[2]]
                tasks.append(pool.apply_async(process_chunk, args=(chunk_idx, chunk)))
                chunk_idx += 1

    pool.close()
    pool.join()

    processed_chunk_indices = [task.get() for task in tasks]
    processed_chunks.update(processed_chunk_indices)

    with open(processed_chunks_path, "wb") as f:
        pickle.dump(processed_chunks, f)
    missing_chunks = set(expected_chunks) - processed_chunks
    if missing_chunks:
        print(f"Warning: Missing chunks detected! {len(missing_chunks)} chunks were not processed.")
        print("Please rerun the script to process missing chunks before merging.")
        return None

    return list(processed_chunks)


def merge_labels(processed_chunks):
    """
    Merges label sets from all processed chunks.
    Ensures all chunks have been processed before merging.
    """
    final_labels = set()
    
    print("Merging extracted labels from chunks...")
    for chunk_idx in tqdm(processed_chunks):
        chunk_file = os.path.join(output_dir, f"labels_chunk_{chunk_idx}.pkl")
        with open(chunk_file, "rb") as f:
            chunk_labels = pickle.load(f)
            final_labels.update(chunk_labels)
    with open(final_output_path, "wb") as f:
        pickle.dump(final_labels, f)

    print(f"Final unique labels saved to {final_output_path}")


if __name__ == "__main__":
    processed_chunks = parallel_process()
    if processed_chunks:
        merge_labels(processed_chunks)
    else:
        print("Merging aborted due to missing chunks. Please rerun the script to complete processing.")

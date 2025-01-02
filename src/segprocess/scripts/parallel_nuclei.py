'''
author: zyx
date: 2025-01-02
description: 
    Merge nuclei segmentation into cell segmentation.
'''
import zarr
import numpy as np
from multiprocessing import Process, JoinableQueue, cpu_count

def downsample_and_merge_chunk(task, cell_zarr_path, nuclei_zarr_path, output_zarr_path):
    '''
    Downsample the cell segmentation, merge with nuclei segmentation, and save the output.
    '''
    index, slices = task

    cell_zarr = zarr.open(cell_zarr_path, mode="r")
    nuclei_zarr = zarr.open(nuclei_zarr_path, mode="r")
    output_zarr = zarr.open(output_zarr_path, mode="a")

    cell_slices = [slice(s.start * 2, s.stop * 2) for s in slices]
    cell_chunk = cell_zarr[tuple(cell_slices)]
    downsampled_cell = cell_chunk[::2, ::2, ::2]

    nuclei_chunk = nuclei_zarr[tuple(slices)]
    output_chunk = np.zeros_like(nuclei_chunk, dtype=np.uint32)

    # Merge logic: if both cell and nuclei are labeled, use cell label.
    mask_both_labeled = (downsampled_cell > 0) & (nuclei_chunk > 0)

    output_chunk[mask_both_labeled] = downsampled_cell[mask_both_labeled]
    output_zarr[tuple(slices)] = output_chunk
    print(f"Processed and merged chunk at index {index}")

def worker(task_queue, cell_zarr_path, nuclei_zarr_path, output_zarr_path):
    while True:
        task = task_queue.get()
        if task is None:
            break
        downsample_and_merge_chunk(task, cell_zarr_path, nuclei_zarr_path, output_zarr_path)
        task_queue.task_done()

def downsample_and_merge_zarr(cell_zarr_path, nuclei_zarr_path, output_zarr_path, num_workers=None):
    cell_zarr = zarr.open(cell_zarr_path, mode="r")
    nuclei_zarr = zarr.open(nuclei_zarr_path, mode="r")
    output_zarr = zarr.open(
        output_zarr_path, mode="w", shape=nuclei_zarr.shape, chunks=nuclei_zarr.chunks, dtype="uint32"
    )

    chunk_shape = nuclei_zarr.chunks
    total_shape = nuclei_zarr.shape
    ndim = nuclei_zarr.ndim

    task_queue = JoinableQueue()
    for chunk_indices in np.ndindex(*[int(np.ceil(total_shape[dim] / chunk_shape[dim])) for dim in range(ndim)]):
        slices = [
            slice(idx * chunk_shape[dim], min((idx + 1) * chunk_shape[dim], total_shape[dim]))
            for dim, idx in enumerate(chunk_indices)
        ]
        task_queue.put((task_queue.qsize(), slices))

    num_workers = num_workers or cpu_count()
    workers = []
    for _ in range(num_workers):
        worker_process = Process(target=worker, args=(task_queue, cell_zarr_path, nuclei_zarr_path, output_zarr_path))
        worker_process.start()
        workers.append(worker_process)

    task_queue.join()

    for _ in range(num_workers):
        task_queue.put(None)
    for worker_process in workers:
        worker_process.join()

    print("Downsampling and merging completed.")

if __name__ == "__main__":
    cell_zarr_path = "/media/zhangy8/ca0155b9-932b-4491-8ea9-d40a586475cf/SegmentationData/CellShape/S2/whole_volume_array_s2_filtered_xyz_20241226.zarr"
    nuclei_zarr_path = "/media/zhangy8/ca0155b9-932b-4491-8ea9-d40a586475cf/SegmentationData/CellShape/S2/whole_volume_nuclei_s3_xyz_20250102.zarr"
    output_zarr_path = "/media/zhangy8/ca0155b9-932b-4491-8ea9-d40a586475cf/SegmentationData/CellShape/S2/whole_volume_nuclei_s3_xyz_20250102_merged_2.zarr"

    downsample_and_merge_zarr(
        cell_zarr_path=cell_zarr_path,
        nuclei_zarr_path=nuclei_zarr_path,
        output_zarr_path=output_zarr_path,
        num_workers=64
    )

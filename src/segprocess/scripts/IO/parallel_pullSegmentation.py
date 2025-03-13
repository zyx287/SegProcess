'''
author: zyx
date: 2025-01-24
last_modified: 2025-01-24
description: 
    Pull segmentation from the ariadne dataset
'''
# TODO: Update the code
import zarr
import numpy as np
from multiprocessing import Process, JoinableQueue, cpu_count
from knossos_utils import KnossosDataset
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

def compute_slices(total_volume_size, chunk_size):
    '''
    Precompute slices for all chunks in the volume.

    Parms:
        total_volume_size: tuple
            The total size of the volume (x, y, z).
        chunk_size: tuple
            The size of each chunk (x, y, z).

    Returns:
        list of tuple
            A list of slices for each chunk.
    '''
    slices_list = []
    for chunk_indices in np.ndindex(*[int(np.ceil(total_volume_size[dim] / chunk_size[dim])) for dim in range(len(total_volume_size))]):
        volume_offset = [idx * chunk_size[dim] for dim, idx in enumerate(chunk_indices)]
        volume_size = [
            min(chunk_size[dim], total_volume_size[dim] - volume_offset[dim])  # Handle boundary cases
            for dim in range(len(chunk_size))
        ]
        slices = [
            slice(volume_offset[dim], volume_offset[dim] + volume_size[dim])
            for dim in range(len(volume_offset))
        ]
        slices_list.append(slices)
    return slices_list

def create_retry_session(retries=5, backoff_factor=0.5):
    """
    Create a requests session with retry logic.
    Parms:
        retries: int
            Number of retry attempts.
        backoff_factor: float
            Backoff factor for retries.
    Returns:
        requests.Session
            A requests session with retry logic.
    """
    session = requests.Session()
    retry = Retry(
        total=retries,
        read=retries,
        connect=retries,
        backoff_factor=backoff_factor,
        status_forcelist=[500, 502, 503, 504],
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session

def chunk_already_processed(output_zarr, downsampled_slices):
    """
    Check if a chunk has already been processed in the Zarr dataset, used for restarting.
    """
    try:
        data = output_zarr[tuple(downsampled_slices)]
        return np.any(data)
    except KeyError:
        return False

def load_and_write_chunk(slices, toml_path, output_zarr_path, mag_size, downsampled_size):
    '''
    Load segmentation data for a specific slice and write to Zarr.

    Parms:
        slices: list of slice
            Precomputed slices defining the chunk.
        toml_path: str
            Path to the TOML file for configuration.
        output_zarr_path: str
            Path to the output Zarr dataset.
        mag_size: int
            Magnification level for segmentation.
        downsampled_size: tuple
            Downsampled size of the dataset.
    '''
    # Compute the offset and size from the slices
    volume_offset = [s.start for s in slices]
    volume_size = [s.stop - s.start for s in slices]
    print(volume_offset, volume_size)

    if chunk_already_processed(output_zarr, downsampled_slices):
        print(f"Skipping already processed chunk: {downsampled_slices}")
        return

    kdataset = KnossosDataset()
    kdataset.initialize_from_conf(toml_path)

    # Avoid ariadne timeout
    session = create_retry_session()
    try:
        # Offset sequence: XYZ, output order: ZYX
        chunk_data = kdataset.load_seg(offset=tuple(volume_offset), size=tuple(volume_size), mag=mag_size)
    except requests.exceptions.RequestException as e:
        print(f"Failed to load segment: {e}")
        raise

    # Convert chunk_data from ZYX to XYZ order :( really weird logic for knossos_utils
    chunk_data = np.transpose(chunk_data, (2, 1, 0))

    downsampled_slices = [
        slice(s.start // mag_size, s.stop // mag_size)
        for s in slices
    ]

    output_zarr = zarr.open(output_zarr_path, mode="a")

    output_zarr[tuple(downsampled_slices)] = chunk_data
    print(f"Processed chunk with offset {volume_offset} and size {volume_size}, written to {downsampled_slices}")

def parse_error_log(error_log_path):
    '''
    Process the error log to get the failed chunks
    '''
    failed_chunks = []
    with open(error_log_path, 'r') as log_file:
        for line in log_file:
            if "Failed to load segment" in line or "Error processing chunk" in line:
                match = re.search(r"\[slice\((\d+), (\d+), None\), slice\((\d+), (\d+), None\), slice\((\d+), (\d+), None\)\]", line)
                if match:
                    slices = [
                        slice(int(match.group(1)), int(match.group(2))),
                        slice(int(match.group(3)), int(match.group(4))),
                        slice(int(match.group(5)), int(match.group(6)))
                    ]
                    failed_chunks.append(slices)
    return failed_chunks


def retry_failed_chunks(error_log_path, toml_path, output_zarr_path, mag_size):
    '''
    Make sure every chunk is processed
    #TODO: Multiprocessing
    '''
    failed_chunks = parse_error_log(error_log_path)
    output_zarr = zarr.open(output_zarr_path, mode="a")

    for slices in failed_chunks:
        try:
            load_and_write_chunk(slices, toml_path, output_zarr_path, mag_size, output_zarr.shape)
        except Exception as e:
            print(f"Failed to retry chunk {slices}: {e}")

def worker(task_queue, toml_path, output_zarr_path, mag_size, downsampled_size):
    """
    Worker function to process tasks from the queue.

    Parms:
        task_queue: JoinableQueue
            Queue containing tasks to process.
        toml_path: str
            Path to the TOML file for configuration.
        output_zarr_path: str
            Path to the output Zarr dataset.
        mag_size: int
            Magnification level for segmentation.
        downsampled_size: tuple
            Downsampled size of the dataset.
    """
   while True:
        slices = task_queue.get()
        if slices is None:
            break
        try:
            load_and_write_chunk(slices, toml_path, output_zarr_path, mag_size, downsampled_size)
        except Exception as e:
            print(f"Error processing chunk {slices}: {e}")
        task_queue.task_done()

def process_volume_with_precomputed_slices(toml_path, output_zarr_path, total_volume_size, chunk_size, mag_size, num_workers=None):
    """
    Process a large volume with precomputed slices and multiprocessing.

    Parms:
        toml_path: str
            Path to the TOML file for configuration.
        output_zarr_path: str
            Path to the output Zarr dataset.
        total_volume_size: tuple
            The total size of the volume (x, y, z).
        chunk_size: tuple
            The size of each chunk (x, y, z).
        mag_size: int
            Magnification level for segmentation.
        num_workers: int, optional
            Number of worker processes to use. Defaults to CPU count.
    """
    downsampled_size = tuple(d // mag_size for d in total_volume_size)
    downsampled_chunk_size = tuple(d // mag_size for d in chunk_size)
    print(f"Downsampled size: {downsampled_size}, Chunk size: {downsampled_chunk_size}")
    zarr.open(output_zarr_path, mode="w", shape=downsampled_size, chunks=downsampled_chunk_size, dtype="uint32")

    # Precompute slices
    slices_list = compute_slices(total_volume_size, chunk_size)
    print(slices_list)

    task_queue = JoinableQueue()
    for slices in slices_list:
        task_queue.put(slices)

    num_workers = num_workers or cpu_count()
    workers = []
    for _ in range(num_workers):
        worker_process = Process(target=worker, args=(task_queue, toml_path, output_zarr_path, mag_size, downsampled_size))
        worker_process.start()
        workers.append(worker_process)

    task_queue.join()

    for _ in range(num_workers):
        task_queue.put(None)
    for worker_process in workers:
        worker_process.join()

    print("Processing completed.")

if __name__ == "__main__":
    toml_path = '/home/zhangy8@hhmi.org/Desktop/2024-08-20 yzhang eviepha9wuinai6EiVujor8Vee2ge8ei.auth.k.toml'
    # For mito (toml_path = '/home/zhangy8@hhmi.org/Desktop/mitos.streaming.k.toml')
    output_zarr_path = "/media/zhangy8/2e9acc20-4fb5-4d17-9e23-9f2ed36f8bf2/SegmentaionData/source/segmentation_data_mag2_20250124.zarr"
    total_volume_size = (27491, 15255, 12548)
    chunk_size = (2048, 2048, 2048)  # Adjust based on memory limit
    mag_size = 2

    process_volume_with_precomputed_slices(
        toml_path=toml_path,
        output_zarr_path=output_zarr_path,
        total_volume_size=total_volume_size,
        chunk_size=chunk_size,
        mag_size=mag_size,
        num_workers=10  # Adjust based on system
    )
    # # Error process
    # error_log_path = "/home/zhangy8@hhmi.org/data1/20250124_segmentation_download/outlog_20250125_v2.log"
    # retry_failed_chunks(error_log_path, toml_path, output_zarr_path, mag_size)
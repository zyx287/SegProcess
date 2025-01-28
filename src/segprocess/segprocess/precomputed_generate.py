'''
author: zyx
date: 2025-01-28
description: 
    Generate precomputed segmentation using igneous and cloudvolume
'''
# TODO: Add from_zarr function to cloudvolume

import os
import time

import numpy as np
import zarr

import dask
from dask.distributed import Client, progress
import multiprocessing
from functools import partial
from cloudvolume import CloudVolume


def generate_from_npy(array, vol_path, **kwargs):
    '''
    Generate precomputed segmentation from numpy array
    '''
    vol = CloudVolume.from_array(
        array,
        vol_path,
        **kwargs
    )
    return vol

def generate_from_zarr(zarr_path, vol_path, layer_type='segmentation',
                       voxel_offset=(0,0,0), resolution=(8,8,8),
                         **kwargs):
    '''
    Generate precomputed segmentation from zarr file
    '''
    zarr_data = zarr.open(zarr_path, mode='r')
    info = CloudVolume.create_new_info(
        num_channels=len(zarr_data.shape), layer_type=layer_type, zarr_data.dtype.name,
        resolution, voxel_offset,
        zarr_data.shape[:3],
        chunk_size=zarr_data.chunks[:3],
        **kwargs
    )
    vol = CloudVolume(
        vol_path,
        info=info,
        bounded=True, compress='br',
        process=False
    )
    vol.commit_info()
    vol.provenance.processing.append({
        'method': 'from zarr',
        'date': time.strftime('%Y-%m-%d %H:%M %Z')
    })
    vol.commit_provenance()
    #TODO Add multiprocess or distributed processing
    for i in range(0, zarr_data.shape[0], zarr_data.chunks[0]):
        for j in range(0, zarr_data.shape[1], zarr_data.chunks[1]):
            for k in range(0, zarr_data.shape[2], zarr_data.chunks[2]):
                chunk = (
                    slice(i, min(i + zarr_data.chunks[0], zarr_data.shape[0])),
                    slice(j, min(j + zarr_data.chunks[1], zarr_data.shape[1])),
                    slice(k, min(k + zarr_data.chunks[2], zarr_data.shape[2]))
                )
                vol[chunk] = zarr_data[chunk]

def transfer_with_dask_distributed(zarr_data, vol):
    tasks = []
    def process_chunk(chunk_slices, zarr_data, vol):
        chunk = zarr_data[chunk_slices]
        vol[chunk_slices] = chunk
    for i in range(0, zarr_data.shape[0], zarr_data.chunks[0]):
        for j in range(0, zarr_data.shape[1], zarr_data.chunks[1]):
            for k in range(0, zarr_data.shape[2], zarr_data.chunks[2]):
                chunk_slices = (
                    slice(i, min(i + zarr_data.chunks[0], zarr_data.shape[0])),
                    slice(j, min(j + zarr_data.chunks[1], zarr_data.shape[1])),
                    slice(k, min(k + zarr_data.chunks[2], zarr_data.shape[2]))
                )
                task = dask.delayed(process_chunk)(chunk_slices, zarr_data, vol)
                tasks.append(task)
    # Trigger the computation with Dask
    dask.compute(*tasks, scheduler="distributed")

def process_chunk_mp(chunk_slices, zarr_data, vol, tracker_file):
    chunk = zarr_data[chunk_slices]
    vol[chunk_slices] = chunk
    with open(tracker_file, "a") as f:
        f.write(f"{chunk_slices}\n")

def transfer_with_multiprocessing(zarr_data, vol, tracker_file, num_workers=4):
    all_chunks = []
    # Iterate over the chunks in the Zarr object
    for i in range(0, zarr_data.shape[0], zarr_data.chunks[0]):
        for j in range(0, zarr_data.shape[1], zarr_data.chunks[1]):
            for k in range(0, zarr_data.shape[2], zarr_data.chunks[2]):
                chunk_slices = (
                    slice(i, min(i + zarr_data.chunks[0], zarr_data.shape[0])),
                    slice(j, min(j + zarr_data.chunks[1], zarr_data.shape[1])),
                    slice(k, min(k + zarr_data.chunks[2], zarr_data.shape[2]))
                )
                all_chunks.append(chunk_slices)
    # Check for already processed chunks
    if os.path.exists(tracker_file):
        with open(tracker_file, "r") as f:
            processed_chunks = {line.strip() for line in f}
    else:
        processed_chunks = set()
    pending_chunks = [chunk for chunk in all_chunks if str(chunk) not in processed_chunks]
    # Use multiprocessing to process chunks
    with multiprocessing.Pool(num_workers) as pool:
        pool.map(
            partial(process_chunk_mp, zarr_data=zarr_data, vol=vol, tracker_file=tracker_file),
            pending_chunks,
        )
    print("All chunks have been processed.")

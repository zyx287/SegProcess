'''
author: zyx
date: 2025-01-28
description: 
    Generate precomputed segmentation using igneous and cloudvolume
'''
# TODO: Add from_zarr function to cloudvolume

import time

import numpy as np
import zarr

import dask
from cloudvolume import CloudVolume
from dask.distributed import Client, progress


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
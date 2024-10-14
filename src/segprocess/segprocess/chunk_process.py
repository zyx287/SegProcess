'''
author: zyx
date: 2024-10-14
description: 
    Functions for processing block data
'''

import os
import numpy as np
import zarr
import dask.array as dda
import xarray as xr
from dask.distributed import Client, LocalCluster

def load_npy_chunk(da, fp, block_info=None, mmap_mode='r'):
    np_mmap = np.load(fp, mmap_mode=mmap_mode)
    array_location = block_info[0]['array-location']
    dim_slicer = tuple(list(map(lambda x: slice(*x), array_location)))
    return np_mmap[dim_slicer]

def dask_read_npy(fp, chunks=None, mmap_mode='r'):
    np_mmap = np.load(fp, mmap_mode=mmap_mode)
    dask_arr = dda.empty_like(np_mmap, chunks=chunks)
    return dask_arr.map_blocks(load_npy_chunk, fp=fp, mmap_mode=mmap_mode, meta=dask_arr)




class BlockArray():
    '''
    Process chunk data
    '''
    def __init__(self, data_path, lookup_path, output_path='.'):
        self.data_path = data_path
        self.lookup_path = lookup_path
        self.output_path = output_path
        
        if not os.path.exists(self.data):
            raise FileNotFoundError(f"File {self.data_path} does not exist")
        if not os.path.exists(self.lookup):
            raise FileNotFoundError(f"File {self.lookup_path} does not exist")
        
        self._chunk_data_load()
        
    def _chunk_data_load(self):
        '''
        Detect the information (parameters, like the size, dtypy, etc.) of given chunk data
        '''
        self.zarr_data = zarr.open(self.data_path, mode='r')
        self.chunk_size = self.zarr_data.chunks
        self.shape = self.zarr_data.shape # For Zarr dataset, the shape format follows (z, y, x)
        self.lookup = self._load_lookup()
    
    def _load_lookup(self):
        '''
        Load the lookup table (A dictionary compressed as a pickle file)
        '''
        import pickle
        with open(self.lookup_path, 'rb') as f:
            data = pickle.load(f)
        return data


    @staticmethod
    def chunk_convert(chunk, lookup):
        '''
        Process chunk data, convert each pixel segnemtation id to the corresponding label
        '''
        converter = np.vectorize(lambda pixel: lookup.get(pixel, 0))
        
        return converter(chunk)
    
    @staticmethod
    def process_chunk(chunk, function, **kwargs):
        '''
        Process chunk data with the given function
        '''
        return function(chunk, **kwargs)

    def simple_process(self, result_dtype=np.uint32):
        '''
        Convert the chunk data to the corresponding label based on the lookup table (sequential processing)
        '''
        output_zarr = zarr.open(self.output_path, mode='w', shape=self.shape, chunks=self.chunk_size, dtype=result_dtype)
        # Process each chunk in zarr data sequentially
        for z in range(0, self.shape[0], self.chunk_size[0]):
            for y in range(0, self.shape[1], self.chunk_size[1]):
                for x in range(0, self.shape[2], self.chunk_size[2]):
                    z_slice = slice(z, min(z + self.chunk_size[0], self.shape[0]))
                    y_slice = slice(y, min(y + self.chunk_size[1], self.shape[1]))
                    x_slice = slice(x, min(x + self.chunk_size[2], self.shape[2]))
                    chunk = self.zarr_data[z_slice, y_slice, x_slice]
                    # Convert chunk
                    print(f"Processing (z, y, x) chunk: {z_slice}, {y_slice}, {x_slice}")
                    processed_chunk = self.chunk_convert(chunk, self.lookup)
                    output_zarr[z_slice, y_slice, x_slice] = processed_chunk
        print("Sequential processing completed.")


    def dask_process(self):
        '''
        Convert the chunk data to the corresponding label based on the lookup table with dask parallel processing
        '''
        pass

    def parallel_process(self):
        '''
        Convert the chunk data to the corresponding label based on the lookup table with dask parallel processing
        '''
        pass




if __name__ == "__main__":
    test_mode = 1
    if test_mode == 1: # test dask functions
        cluster = LocalCluster()
        client = Client(cluster)
        npy_file = ''
        chunks = (64, 64, 64)
        dask_array = dask_read_npy(npy_file, chunks=chunks, mmap_mode='r')
        xarray_data = xr.DataArray(dask_array, dims=['x', 'y', 'z'])
        zarr_output_path = '' # File path of the output dataset
        xarray_data.to_zarr(zarr_output_path, mode='w')
        print(f"Data has been successfully written to {zarr_output_path}")
    elif test_mode == 2: # test block processing
        block_array = BlockArray(data_path='input_data.zarr', lookup_path='lookup.npz', output_path='output_data.zarr')
        block_array.simple_process()
        print("Block processing completed.")
        block_array.dask_process()
        print("Block processing completed with dask.")
        block_array/.parallel_process()
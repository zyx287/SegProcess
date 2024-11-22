'''
author: zyx
date: 2024-10-15
description: 
    Generate dataset for proofread cell
'''

import os
import datetime
import pandas as pd
import pickle
import numpy as np
import zarr
import dask
import dask.array as da
from dask import delayed
from segprocess.segprocess.knossos_utils_plugin import (
    load_knossos_dataset,
    generate_segmentation,
    seg_to_tif,
    launch_kimimaro
)
from segprocess.segprocess.graph_reader import SegGraph
from segprocess.segprocess.knossos_utils_plugin import load_knossos_dataset


class ProofDataset():
    def __init__(self, excel_path='/Users/zyx/Desktop/000LiLab/ariadne proofread track/20240929/EmeraldProofreading_external_20240929.xlsx'):
        self.excel_path = excel_path

    def filter_excel_client(self, sheets=['zone 1', 'zone 2', 'zone 3']):
        '''
        Read and select the cells after proofreading and marked as 'complete'.
        '''
        file_path = self.excel_path
        self.sheets = sheets
        combined_data = []

        for sheet in sheets:
            df = pd.read_excel(file_path, sheet_name=sheet)
            filtered_df = df[df['client status'] == 'complete']
            filtered_df['Zone info'] = sheet
            for _, row in filtered_df.iterrows():
                zone_data = (
                    row['redmine number'], 
                    row['Cell Name'], 
                    row['coordinate'], 
                    row['client status'], 
                    row['Zone info']
                )
            combined_data.append(zone_data)
        
        combined_df = pd.DataFrame(combined_data, columns=['Redmine Number', 'Cell Name', 'Coordinate', 'Client Status', 'Zone Info'])
        self.processed_df = combined_df

    def filter_excel_ariadne(self):
        '''
        Read and select the cells after proofreading and marked as 'QA is complete'.
        '''
        file_path = self.excel_path
        sheets = ['zone 1', 'zone 2', 'zone 3']
        combined_data = []

        for sheet in sheets:
            df = pd.read_excel(file_path, sheet_name=sheet)
            filtered_df = df[df['ariadne status'] == 'QA is completed']
            filtered_df['Zone info'] = sheet
            for _, row in filtered_df.iterrows():
                zone_data = (
                    row['redmine number'], 
                    row['Cell Name'], 
                    row['coordinate'], 
                    row['ariadne status'], 
                    row['Zone info']
                )
                combined_data.append(zone_data)
        
        combined_df = pd.DataFrame(combined_data, columns=['Redmine Number', 'Cell Name', 'Coordinate', 'ariadne status', 'Zone Info'])
        self.processed_df = combined_df
    
    @property
    def get_processed_df(self):
        return self.processed_df

    def save_new_dataset(self):
        '''
        Save the processed data to a new Excel file.
        '''
        timestamp = datetime.datetime.now().strftime('%Y%m%d')
        save_path = self.excel_path.replace('.xlsx', f'_{timestamp}processed.csv')
        self.processed_df.to_csv(save_path, index=False)
        print(f"Processed data saved at {save_path}")

    @staticmethod
    def _map_coordinates(coord):
        return tuple(map(int, coord.split(',')))
    
    @staticmethod
    def _downsampling(coord, factor):
        return (tuple(c//factor for c in coord))

    
    def get_one_pixel_seg_id(self,
                             toml_file_path='/home/zhangy8@hhmi.org/Desktop/2024-08-20 yzhang eviepha9wuinai6EiVujor8Vee2ge8ei.auth.k.toml',
                             lookup_path='/home/zhangy8@hhmi.org/data1/pipeline_test/Segmentation_generation/notebooks/id_to_label_202409.pkl'
                             ):
        '''
        Get the segmentation id for all neurons in the list.
        '''
        self.processed_df['Mapped_Coordinates'] = self.processed_df['Coordinate'].apply(self._map_coordinates)
        target_list = [
            load_knossos_dataset(
                toml_path=toml_file_path,
                volume_offset=coord,
                volume_size=(32, 32, 32),
                mag_size=5
            )[0][0][0]
            for coord in self.processed_df['Mapped_Coordinates']
        ]
        self.processed_df['target_id'] = target_list

        with open(lookup_path, 'rb') as f:
            lookup_data = pickle.load(f)
        self.processed_df['label'] = self.processed_df['target_id'].apply(
            lambda seg_id: int(float(lookup_data.get(seg_id, 'Segmentation id not found')))
        )
        print("Segmentation id and label generated for all neurons.")
    
    def get_downsampling_seg_id(self, downsampling_factors=[2, 3, 4, 5]):
        for factor in downsampling_factors:
            self.processed_df[f'Downsampled_Coordinate_{factor}'] = self.processed_df['Mapped_Coordinates'].apply(
                lambda coord: self._downsampling(coord, factor))
        print("Downsampled coordinates generated for all neurons.")

class SegmentationData():
    '''
    A class to generate segmentation data array use knossos_utils chunkwise
    '''
    def __init__(self, output_dir='./segmentation_data'):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
    
    def fetch_segmentation_chunk(self, chunk_size, offset, mag):
        '''
        Get segmentation chunk based on knossos_utils_plugin
        '''
        chunk_size_xyx = chunk_size[::-1]  # Convert z, y, x to x, y, z for knossos_utils
        chunk_array = load_knossos_dataset(volume_offset=offset, volume_size=chunk_size_xyx, mag_size=mag)
        return chunk_array.astype(np.int32)
    
    def generate_segmentation(self, chunk_size, mag, volume_size):
        '''
        Generate the entire volume segmentation chunkwise and save to Zarr.
        Params:
            chunk_size: tuple of int (z, y, x)
                The size of each chunk to download.
            mag: int
                The magnification level.
            volume_size: tuple of int (z, y, x)
                The total volume size.
        '''
        zarr_path = os.path.join(self.output_dir, 'segmentation_data.zarr')
        zarr_array = zarr.open(
            zarr_path,
            mode='w',
            shape=volume_size,
            chunks=chunk_size,
            dtype=np.int32
        )
        
        for z in range(0, volume_size[0], chunk_size[0]):
            for y in range(0, volume_size[1], chunk_size[1]):
                for x in range(0, volume_size[2], chunk_size[2]):
                    offset = (x, y, z)  # Offset in x, y, z order for knossos_utils
                    current_chunk_size = tuple(
                        min(chunk_size[i], volume_size[i] - [z, y, x][i])
                        for i in range(3)
                    )
                    chunk_data = self.fetch_segmentation_chunk(current_chunk_size, offset, mag)
                    zarr_array[z:z+current_chunk_size[0],
                               y:y+current_chunk_size[1],
                               x:x+current_chunk_size[2]] = chunk_data
                    print(f"Chunk saved at offset {offset} with shape {current_chunk_size}")
        
        print(f"Segmentation data saved to {zarr_path}")

    def generate_segmentation_parallel(self, chunk_size, mag, volume_size):
        '''
        Generate the entire volume segmentation chunk-by-chunk in parallel and save to Zarr.
        Params:
            chunk_size: tuple of int (z, y, x)
                The size of each chunk to download.
            mag: int
                The magnification level.
            volume_size: tuple of int (z, y, x)
                The total volume size.
        '''
        zarr_path = os.path.join(self.output_dir, 'segmentation_data_para.zarr')
        zarr_array = zarr.open(
            zarr_path,
            mode='w',
            shape=volume_size,
            chunks=chunk_size,
            dtype=np.int32
        )
        
        tasks = []
        for z in range(0, volume_size[0], chunk_size[0]):
            for y in range(0, volume_size[1], chunk_size[1]):
                for x in range(0, volume_size[2], chunk_size[2]):
                    offset = (x, y, z)  # Offset in x, y, z order for knossos_utils
                    current_chunk_size = tuple(
                        min(chunk_size[i], volume_size[i] - [z, y, x][i])
                        for i in range(3)
                    )
                    # Create a delayed task for each chunk
                    task = delayed(self.process_and_store_chunk)(
                        zarr_array, current_chunk_size, offset, mag
                    )
                    tasks.append(task)
        
        # Compute all tasks in parallel
        dask.compute(*tasks)
        print(f"Segmentation data saved to {zarr_path}")
    
    def process_and_store_chunk(self, zarr_array, chunk_size, offset, mag):
        '''
        Fetch a chunk and store it into the Zarr array at the specified offset.
        Params:
            zarr_array: Zarr array
                The target Zarr array to store data.
            chunk_size: tuple of int (z, y, x)
                The actual size of the chunk to fetch and store.
            offset: tuple of int (x, y, z)
                The offset where the chunk should be placed in the Zarr array.
            mag: int
                The magnification level.
        '''
        chunk_data = self.fetch_segmentation_chunk(chunk_size, offset, mag)
        z, y, x = offset[2], offset[1], offset[0]  # Convert offset to z, y, x for Zarr indexing
        zarr_array[z:z+chunk_size[0], y:y+chunk_size[1], x:x+chunk_size[2]] = chunk_data
        print(f"Chunk saved at offset {offset} with shape {chunk_size}")

    def fetch_segmentation_slice(self, z, xy_size, mag):
        '''
        Fetch a single z-slice of the segmentation based on knossos_utils.
        '''
        offset = (0, 0, z)
        slice_size = (xy_size[1], xy_size[0], z)  # (x, y, z), where z=1 for a single slice
        slice_array = load_knossos_dataset(volume_offset=offset, volume_size=slice_size, mag_size=mag)
        return slice_array[0].astype(np.int32)  # Extract the 2D (y, x) part of the slice
    
    def generate_segmentation_slice(self, xy_size, mag, total_z):
        '''
        Generate the entire volume by fetching one z-slice at a time and saving to Zarr.
        Params:
            xy_size: tuple of int (y, x)
                The size of each xy slice.
            mag: int
                The magnification level.
            total_z: int
                The total number of z slices in the volume.
        '''
        # Create the Zarr array with the full volume shape
        volume_shape = (total_z, xy_size[0], xy_size[1])
        zarr_path = os.path.join(self.output_dir, 'segmentation_data_zslice.zarr')
        zarr_array = zarr.open(
            zarr_path,
            mode='w',
            shape=volume_shape,
            chunks=(1, xy_size[0], xy_size[1]),
            dtype=np.int32  # Ensure the correct data type
        )
        
        for z in range(total_z):
            slice_data = self.fetch_segmentation_slice(z, xy_size, mag)
            zarr_array[z, :, :] = slice_data
            print(f"Slice saved at z={z} with shape {slice_data.shape}")
        print(f"Segmentation data saved to {zarr_path}")

    def generate_segmentation_zslice_parallel(self, xy_size, mag, total_z):
        '''
        Generate the entire volume in parallel by fetching one z-slice at a time and saving to Zarr.
        Params:
            xy_size: tuple of int (y, x)
                The size of each xy slice.
            mag: int
                The magnification level.
            total_z: int
                The total number of z slices in the volume.
        '''
        volume_shape = (total_z, xy_size[0], xy_size[1])
        zarr_path = os.path.join(self.output_dir, 'segmentation_data_z_slice_para.zarr')
        zarr_array = zarr.open(
            zarr_path,
            mode='w',
            shape=volume_shape,
            chunks=(1, xy_size[0], xy_size[1]),
            dtype=np.int32
        )
        
        tasks = []
        for z in range(total_z):
            # Delayed task to fetch and store each z-slice
            task = delayed(self.process_and_store_slice)(zarr_array, z, xy_size, mag)
            tasks.append(task)
        dask.compute(*tasks)
        print(f"Segmentation data saved to {zarr_path}")
    
    def process_and_store_slice(self, zarr_array, z, xy_size, mag):
        '''
        Fetch a single z-slice and store it into the Zarr array.
        Params:
            zarr_array: Zarr array
                The target Zarr array to store data.
            z: int
                The z index where the slice should be placed in the Zarr array.
            xy_size: tuple of int (y, x)
                The size of each xy slice.
            mag: int
                The magnification level.
        '''
        slice_data = self.fetch_segmentation_slice(z, xy_size, mag)
        zarr_array[z, :, :] = slice_data
        print(f"Slice saved at z={z} with shape {slice_data.shape}")
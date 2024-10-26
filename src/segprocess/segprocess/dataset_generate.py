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
from segprocess.segprocess.knossos_utils_plugin import (
    load_knossos_dataset,
    generate_segmentation,
    seg_to_tif,
    launch_kimimaro
)
from segprocess.segprocess.graph_reader import SegGraph


class ProofDataset():
    def __init__(self, excel_path='/Users/zyx/Desktop/000LiLab/ariadne proofread track/20240929/EmeraldProofreading_external_20240929.xlsx'):
        self.excel_path = excel_path

    def filter_excel_client(self):
        '''
        Read and select the cells after proofreading and marked as 'complete'.
        '''
        file_path = self.excel_path
        sheets = ['zone 1', 'zone 2', 'zone 3']
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
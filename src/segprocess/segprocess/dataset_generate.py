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
from segprocess.knossos_utils_plugin import (
    load_knossos_dataset,
    generate_segmentation,
    seg_to_tif,
    launch_kimimaro
)
from segprocess.graph_reader import SegGraph


class ProofDataset():
    def __init__(self, excel_path):
        self.excel_path = excel_path
        self.filter_excel()

    def filter_excel(self):
        '''
        Read and select the cells after proofreading and marked as 'complete'.
        '''
        file_path = '/Users/zyx/Desktop/000LiLab/ariadne proofread track/20240929/EmeraldProofreading_external_20240929.xlsx'  # Update with your actual file path
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
        a, b, c = map(int, coord.split(','))
        return (a, b, c)
    
    def get_one_pixel_seg_id(self,
                             toml_file_path='/home/zhangy8@hhmi.org/Desktop/2024-08-20 yzhang eviepha9wuinai6EiVujor8Vee2ge8ei.auth.k.toml',
                             lookup_path='/home/zhangy8@hhmi.org/data1/pipeline_test/Segmentation_generation/notebooks/id_to_label_202409.pkl'
                             ):
        '''
        Get the segmentation id for all neurons in the list.
        '''
        map_ids = []
        for _, row in self.processed_df.iterrows():
            coord_str = row['Coordinate']
            # Generate the mapped ID for the coordinate
            map_id = self._map_coordinates(coord_str)
            map_ids.append(map_id)
        target_list = []
        for num, coor in enumerate(map_ids):
            one_volume_array = load_knossos_dataset(toml_path=toml_file_path,
                                                    volume_offset=coor,
                                                    volume_size=(32, 32, 32),
                                                    mag_size=5)  
            target_list.append(one_volume_array[0][0][0])
            del one_volume_array
        self.processed_df['target_id'] = target_list
        with open(lookup_path, 'rb') as f:
            data = pickle.load(f)
        label_list = []
        for id in target_list:
            if id not in data.keys():
                print(f"Segmentation id {id} not found in the lookup table.")
            label_list.append(int(float(data[id])))
        self.processed_df['label'] = label_list
        print("Segmentation id and label generated for all neurons.")
        
'''
author: zyx
date: 2024-09-11
description: 
    scripts for processing ariadne.ai data
    - Generate segmentation based on Knossos_utils and mergelist (exported from KNOSSOS)
    - Skeletonization of the segmentation based on Kimimaro
'''

import os
import numpy as np
import pandas as pd

import tifffile as tiff
from knossos_utils.knossosdataset import KnossosDataset

def load_knossos_dataset(toml_path: str = "./toml-file/2024-08-20 yzhang eviepha9wuinai6EiVujor8Vee2ge8ei.auth.k.toml",
                         volume_offset: tuple = (10000, 0, 3000),
                         volume_size: tuple = (4000, 7000, 9000),
                         mag_size: int = 4):
    '''
    Follow instructions from ariadne.ai
    Loading toml -> load_seg
    '''
    kdataset = KnossosDataset()
    kdataset.initialize_from_conf(toml_path)
    around_volume = kdataset.load_seg(offset=volume_offset, size=volume_size, mag=mag_size)
    return around_volume

def generate_segmentation(segmented_volume: np.ndarray, mergelist_path: str = 'mergelist.txt'):
    '''
    Generate segmentation based on Knossos_utils and mergelist (exported from KNOSSOS)
    '''
    with open(mergelist_path, 'r') as file:
        file_contents = file.read()
        number_set = set(map(int, file_contents.split()))# Speed up using searching based on set
    
    mask_segmention =  np.array([1 if ((elem in number_set) and (elem != 0)) else 0 for elem in segmented_volume.flatten()])
    mask_segmention_reshaped = mask_segmention.reshape(segmented_volume.shape)
    return mask_segmention_reshaped

def seg_to_tif(segmentation: np.ndarray, save_path: str = './segmentation.tif'):
    '''
    Save segmentation to tif file
    '''
    segmentation_tif = (segmentation * 255).astype(np.uint8)
    tiff.imwrite(save_path, segmentation_tif)

def seg_to_npy(segmentation: np.ndarray, save_path: str = './segmentation.npy'):
    '''
    Save segmentation to tif file
    '''
    np.save(save_path, segmentation)

def generate_npy_from_uint8img(path):
    '''
    Save tiff file to npy file
    '''
    segment = tiff.imread(path)
    segment = segment.astype(np.uint8)
    np.save(path.replace('.tif', '.npy'), np.array(segment))

def launch_kimimaro(file_path: str, env_name: str = 'kimimaro_env')-> int:
    '''
    Launch Kimimaro for skeletonization
        requirment: install kimimaro package in a conda environment
    '''
    conda_commend = f"conda activate {env_name}"
    os.system(conda_commend)

    kimimaro_commend = f"kimimaro forge {file_path} --progress"
    os.system(kimimaro_commend)
    
    return 0
'''
author: zyx
date: 2024-09-20
description: 
    A pipeline for generate segmentation data based on ariadne outpit
'''

# main_pipeline.py

import os
from segprocess.segprocess.knossos_utils_plugin import (
    load_knossos_dataset,
    generate_segmentation,
    seg_to_tif,
    launch_kimimaro
)
from segprocess.segprocess.graph_reader import SegGraph

import pandas as pd
import numpy as np
import argparse

########### Helper functions ###########
def map_coordinates(coord):
    a, b, c = map(int, coord.split(','))
    return (a, b, c)

def get_parse():
    parse = argparse.ArgumentParser(description='Batch process for generating segmentation data based on ariadne output')
    parse.add_argument('--coordinate_dataframe_path', type=str, help='Path to the coordinate dataframe')
    parse.add_argument('--toml_path', type=str, help='Path to the toml file')
    parse.add_argument('--graph_path', type=str, help='Path to the graph file')
    parse.add_argument('--graph_changes_path', type=str, help='Path to the graph changes file')
    parse.add_argument('--output_dir_path', type=str, help='Path to the output directory')
    parse.add_argument('--mag_size', type=int, default=5, help='Magnification size')
    return parse.parse_args()
########### Helper functions ###########

def generate_mergelist(
    coordinate_dataframe_path: str,
    toml_path: str,
    graph_path: str,
    graph_changes_path: str,
    output_dir_path: str,
    mag_size: int = 5
):
    if not os.path.exists(output_dir_path):
        os.makedirs(output_dir_path)
    else:
        raise ValueError(f"Output directory already exists: {output_dir_path}")

    coor_data_df = pd.read_csv(coordinate_dataframe_path, index_col=0)
    # Load Knossos Graph
    ariadne_graph = SegGraph(graph_path=graph_path)
    ariadne_graph.update_graph_changes(graph_changes_path)
    # Generate target_id list
    map_ids = []
    for _, row in coor_data_df.iterrows():
        coord_str = row['Coordinate']
        # Generate the mapped ID for the coordinate
        map_id = map_coordinates(coord_str)
        map_ids.append(map_id)
    target_list = []
    for num, coor in enumerate(map_ids):
        one_volume_array = load_knossos_dataset(toml_path=toml_path,
                                                volume_offset=coor,
                                                volume_size=(16, 16, 16),
                                                mag_size=5)  
        target_list.append(one_volume_array[0][0][0])
        del one_volume_array
    coor_data_df['target_id'] = target_list
    coor_data_df.to_csv(os.path.join(output_dir_path, 'coordinate_target_id.csv'))
    # process mergelist
    process_list = []
    start_num = 0
    for _, row in coor_data_df.iterrows():
        target_str = row['target_id']
        cell_id = coor_data_df.index[start_num]
        process_list.append([cell_id, target_str])
        start_num += 1
    ## Generate mergelist
    for num, info in enumerate(process_list):
        cell_id = info[0]
        target_id = info[1]
        if not os.path.exists(os.path.join(output_dir_path, 'mergelistsum')):
            os.makedirs(os.path.join(output_dir_path, 'mergelistsum'))
        if not os.path.exists(os.path.join(os.path.join(output_dir_path, 'mergelistsum'), cell_id)):
            os.makedirs(os.path.join(os.path.join(output_dir_path, 'mergelistsum'), cell_id))
        merge_set = ariadne_graph.extract_connected_components(target_id)
        # saving merge_set into txt file
        with open(os.path.join(os.path.join(os.path.join(output_dir_path, 'mergelistsum'), cell_id),'mergelist.txt'), 'w') as f:
            for item in list(merge_set):
                f.write("%s " % item)
        del merge_set
        print(f'{cell_id} is done')

def main():
    args = get_parse()
    generate_mergelist(
        coordinate_dataframe_path=args.coordinate_dataframe_path,
        toml_path=args.toml_path,
        graph_path=args.graph_path,
        graph_changes_path=args.graph_changes_path,
        output_dir_path=args.output_dir_path,
        mag_size=args.mag_size
    )



    # Define parameters
    coordinates_list = [
        # List of coordinates as tuples, e.g.,
        # (x1, y1, z1),
        # (x2, y2, z2),
    ]
    toml_path = "."
    mergelist_path = "mergelist.txt"
    output_dir = "./output"
    volume_size = (4000, 7000, 9000)  # Adjust as needed
    mag_size = 4  # Adjust as needed

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Process each coordinate
    for coordinate in coordinates_list:
        try:
            process_coordinate(
                coordinate,
                toml_path,
                mergelist_path,
                output_dir,
                volume_size,
                mag_size
            )
            print(f"Processed coordinate: {coordinate}")
        except Exception as e:
            print(f"Error processing coordinate {coordinate}: {e}")

if __name__ == "__main__":
    main()

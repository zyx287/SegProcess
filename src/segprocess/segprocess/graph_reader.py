'''
author: zyx
date: 2024-09-17
description: 
    Python scripts for loading and modifying segmentation graph data
Details:
A class requires instantiating for process and output the segmentation graph data.
Useful functions:
1. updata_graph_changes:
    The segmentation changes were updated by ariadne, which could be directly "wget" based on the given URL.
    The changes were applied to the base graph, and could be save as a new graph based on save_graph.
    This method should be called after instantiation.
    e.g. seg_graph = SegGraph(path) -> seg_graph.update_graph_changes(change_log_path)
2. extract_connected_components:
    This method is used for generating the merge list of a specific neuron.
    The connected components of the given supervoxel were extracted from the base graph, which contains all the supervoxels belong to the given cell.
    After instantiating the class, this method could be called directly.
    For example, seg_graph = SegGraph(path) -> seg_graph.extract_connected_components(start_vertex_supervoxel_id)
    Usually, the updated graph should be called before this method.
3. generate_supervoxel_label_dirc and get_supervoxel_to_label:
    This method is used for generating the {supervoxel: label} dictionary, which is used for processing the whole volume segnegmentation.
    Also, it should be called after the instantiation and update_graph_changes.
    e.g. seg_graph = SegGraph(path) -> ... -> seg_graph.generate_supervoxel_label_dirc()
    Then, utilize the get_supervoxel_to_label to get the {supervoxel: label} dictionary, which is a read-only method.
'''

import os
from datetime import datetime
import numpy as np
import pandas as pd
import graph_tool.all as gt

class SegGraph():
    '''
    Class for processing the graph for segmentation
    '''
    def __init__(self, graph_path: str):
        '''
        Loading graph from the given path
        '''
        self.base_graph = gt.load_graph(graph_path)
        self.seg_id_to_vertex, self.vertex_to_segid = self._load_base_graph_segids()

        self.supervoxel_to_label = {}

    def _load_base_graph_segids(self)->dict:
        '''
        Load the segmentation ids from the base graph
        '''
        seg_id_group = self.base_graph.vp['seg_ids']
        seg_id_to_vertex = {}
        vertex_to_seg_id = {}
        for v in self.base_graph.vertices():
            seg_id_to_vertex[int(seg_id_group[v])] = v
            vertex_to_seg_id[int(v)] = seg_id_group[v]

        return seg_id_to_vertex, vertex_to_seg_id

    def get_latest_changes(self, changes_path = 'https://emerald.agglomeration.ariadne.ai/api/changes.tsv'):
        import os
        current_path = os.getcwd()
        download_commend = f"wget {changes_path}"
        try:
            os.system(download_commend)
            print(f"Download successfully! Changes file saved as {current_path}/changes.tsv")
        except:
            print("Download failed! Please check the URL.")
        return 0
        

    def update_graph_changes(self, change_log_path: str):
        '''
        Update the graph according to the change log
        '''
        changes = pd.read_csv(change_log_path, sep='\t',
                              header=None,
                              names=['operation', 'vertex1', 'vertex2']
        )

        # Apply changes to the base graph
        # operation_num = 0
        for index, row in changes.iterrows():
            operation = row['operation']
            seg_id_1 = row['vertex1']
            seg_id_2 = row['vertex2']
            
            if seg_id_1 in self.seg_id_to_vertex and seg_id_2 in self.seg_id_to_vertex:
                vertex1 = self.seg_id_to_vertex[seg_id_1]
                vertex2 = self.seg_id_to_vertex[seg_id_2]
                if operation == '+':
                    self.base_graph.add_edge(vertex1, vertex2)
                    # operation_num += 1
                elif operation == '-':
                    edge = self.base_graph.edge(vertex1, vertex2)
                    if edge is not None:
                        self.base_graph.remove_edge(edge)
                    # operation_num += 1
            else:
                raise ValueError(f"Changes {index} error! Segmentation id {seg_id_1} or {seg_id_2} not found in the base graph")
    
    def extract_connected_components(self, start_vertex_supervoxel_id: int)-> int:
        '''
        Extract all connected components of the given supervoxel from the base graph (Works for generating merge list of a specific neuron)
        '''
        if start_vertex_supervoxel_id not in self.seg_id_to_vertex:
            raise ValueError(f"Supervoxel {start_vertex_supervoxel_id} not found in the base graph")
        start_vertex = self.seg_id_to_vertex[start_vertex_supervoxel_id]
        print(f"seg_id {start_vertex_supervoxel_id}: {int(start_vertex)}")
        visited_vertices_segid = set()
        for edge in gt.dfs_iterator(self.base_graph, start_vertex):
            source_vertex = int(edge.source())
            target_vertex = int(edge.target())
            
            visited_vertices_segid.add(self.vertex_to_segid[int(source_vertex)])
            visited_vertices_segid.add(self.vertex_to_segid[int(target_vertex)])
        
        return visited_vertices_segid
    
    def generate_supervoxel_label_dirc(self, transform_flag=True):
        '''
        Generate the {supervoxel: label} dictionary
        '''
        vertex_to_label = {}
        comp, hist = gt.label_components(self.base_graph)
        for v in self.base_graph.vertices():
            vertex_to_label[int(v)] = comp[v]
        # Map two dictionaries
        supervoxel_to_label = {}
        for sv_id, v in self.seg_id_to_vertex.items():
            label = vertex_to_label[int(v)]
            supervoxel_to_label[sv_id] = vertex_to_label[v]
        
        if not transform_flag:
            self.supervoxel_to_label = supervoxel_to_label
            return 0
        ## Simple transform the data:
        label_to_new_label = {}
        start_number = 9 * (10 ** (int(len(str(len(hist))))))
        for label in range(len(hist)):
            label_to_new_label[label] = start_number + label
        supervoxel_to_new_label = {}
        for sv_id, label in supervoxel_to_label.items():
            new_label = label_to_new_label[label]
            supervoxel_to_new_label[sv_id] = new_label
        self.supervoxel_to_label = supervoxel_to_new_label
        return 1

    @property
    def get_supervoxel_to_label(self):
        return self.supervoxel_to_label
    
    def save_graph(self, output_file_path: str):
        '''
        Save the graph to the output path
        '''
        self.base_graph.save(output_file_path)
        folder_path = os.path.dirname(output_file_path)
        current_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        date_file_path = os.path.join(folder_path, "save_date_info")
        with open(date_file_path, 'w') as date_file:
            date_file.write(f"Graph saved on: {current_date}")

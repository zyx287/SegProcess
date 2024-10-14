'''
Read files from the input directory and convert them to the output directory at different formats
'''

import os
from abc import ABC, abstractmethod
from SegProcess.SegProcess import mesh_convert, skeleton_convert, object_convert

class Converter(ABC):
    '''
    An abstract class for file conversion, with the classmethod convert
    '''
    def __init__(self, file_path: str, output_folder_path: str):
        self.file_path = file_path
        self.file_type = self.detertmine_file_type()
        self.output_folder_path = output_folder_path

    def detertmine_file_type(self):
        file_extension = os.path.splitext(self.file_path)[-1]
        return file_extension.replace('.', '')
    
    @abstractmethod
    def convert(self)->str:
        pass

class MeshConverter(Converter):
    def convert(self):
        if self.file_type == "stl":
            return mesh_convert.stl_to_mesh(self.file_path, self.output_folder_path)
        elif self.file_type == "vtk":
            return mesh_convert.vtk_to_npy(self.file_path, self.output_folder_path)
        elif self.file_type == "npy":
            return mesh_convert.loadmesh_trimesh(self.file_path)
        else:
            raise ValueError(f"File type {self.file_type} not supported for mesh conversion")
        
class SkeletonConverter(Converter):
    def convert(self):
        if self.file_type == "npy":
            return skeleton_convert.npy_to_swc(self.file_path, self.output_folder_path)
        elif self.file_type == "hdf5":
            return skeleton_convert.hdf5_to_swc(self.file_path, self.output_folder_path)
        elif self.file_type == "vtk":
            print("WARNING: VTK to SWC conversion may not be accurate")
            return skeleton_convert.vtk_to_swc(self.file_path, self.output_folder_path)
        else:
            raise ValueError(f"File type {self.file_type} not supported for skeleton conversion")

class ObjectConverter(Converter):
    def convert(self):
        if self.file_type == "npy":
            return object_convert.npy_to_vtk(self.file_path, self.output_folder_path)
        elif self.file_type == "hdf5":
            return object_convert.hdf5_to_vtk(self.file_path, self.output_folder_path)
        elif self.file_type == "swc":
            return object_convert.swc_to_vtk(self.file_path, self.output_folder_path)
        else:
            raise ValueError(f"File type {self.file_type} not supported for object conversion")


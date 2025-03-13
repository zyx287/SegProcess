import numpy as np
import zarr
from cloudvolume import CloudVolume
zarr_path = "/media/zhangy8/ca0155b9-932b-4491-8ea9-d40a586475cf/SegmentationData/CellShape/S1/cellshape_s1_filtered_20250305_xyz_new.zarr"
zarr_data = zarr.open(zarr_path, mode='r')

info = CloudVolume.create_new_info(
     1, 'segmentation', 'uint32',
      'compresso', [16,16,16],
      [0,0,0], (13745, 7627, 6274),
      chunk_size=(256,256,256), max_mip=0
    )
vol = CloudVolume(
    'file:///media/zhangy8/ca0155b9-932b-4491-8ea9-d40a586475cf/SegmentationData/CellShape/cell_s1_igneous_20250305_new',
    info=info, bounded=True, 
    compress='br', progress=False
)
import sys
import time
vol.commit_info()
vol.provenance.processing.append({
    'method': 'from zarr',
    'date': time.strftime('%Y-%m-%d %H:%M %Z')
})
vol.commit_provenance()

import zarr
import numpy as np
# Iterate over the chunks
for i in range(0, zarr_data.shape[0], zarr_data.chunks[0]):
    for j in range(0, zarr_data.shape[1], zarr_data.chunks[1]):
        for k in range(0, zarr_data.shape[2], zarr_data.chunks[2]):
            # Compute the chunk range with boundary handling
            chunk = (
                slice(i, min(i + zarr_data.chunks[0], zarr_data.shape[0])),
                slice(j, min(j + zarr_data.chunks[1], zarr_data.shape[1])),
                slice(k, min(k + zarr_data.chunks[2], zarr_data.shape[2]))
            )
            
            # Read the chunk from the source and write it to the target
            vol[chunk] = zarr_data[chunk]

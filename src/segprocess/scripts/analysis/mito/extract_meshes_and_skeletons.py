'''
author: zyx
date: 2025-02-10
last modified: 2025-02-10
description: 
    Step2: Extract meshes and skeletons from the mitochondria segmentation
'''
import os
import pickle
import multiprocessing
import trimesh
from cloudvolume import CloudVolume
from tqdm import tqdm

# Define paths
volume_path = "file:///media/zhangy8/ca0155b9-932b-4491-8ea9-d40a586475cf/SegmentationData/CellShape/mitochondria_250208_mag2.zarr/"
output_ply_dir = "/media/zhangy8/ca0155b9-932b-4491-8ea9-d40a586475cf/SegmentationData/Organelle/Mitochondria/Mesh/ply/"
output_xyz_dir = "/media/zhangy8/ca0155b9-932b-4491-8ea9-d40a586475cf/SegmentationData/Organelle/Mitochondria/Mesh/xyz/"
output_skeleton_dir = "/media/zhangy8/ca0155b9-932b-4491-8ea9-d40a586475cf/SegmentationData/Organelle/Mitochondria/Skeleton/"
processed_labels_path = "/home/zhangy8@hhmi.org/data1/20250210_assign_mitochondria/Output/final/mesh_processed_label.pkl" # path to save processed labels (only for step2)
final_output_path = "/home/zhangy8@hhmi.org/data1/20250210_assign_mitochondria/Output/final/final(mesh_skeleton)_labels.pkl" # path to save final processed labels, also only for step2

os.makedirs(output_ply_dir, exist_ok=True)
os.makedirs(output_xyz_dir, exist_ok=True)


with open("/home/zhangy8@hhmi.org/data1/20250210_assign_mitochondria/Output/final/unique_labels.pkl", "rb") as f:
    all_labels = pickle.load(f)
print(f"Total unique labels: {len(all_labels)}")
num_workers = 80

def process_label(label):
    """
    Extracts mesh for a given label and saves PLY & XYZ files.
    """
    if label == 0:
        return label  # Skip label 0

    ply_path = os.path.join(output_ply_dir, f"{label}_mito_s1.ply")
    xyz_path = os.path.join(output_xyz_dir, f"{label}_mito_s1.xyz")
    # skeleton_path = os.path.join(output_skeleton_dir, f"{label}_skeleton.swc")

    if os.path.exists(ply_path) and os.path.exists(xyz_path):
        print(f"Mesh already exists for label {label}.")
        return label
    try:
        vol = CloudVolume(volume_path, mip=0)
        mesh = vol.mesh.get(label)
        with open(ply_path, "wb") as f:
            f.write(mesh.to_ply())

        mesh_2 = trimesh.load_mesh(ply_path)
        xyz_data = trimesh.exchange.xyz.export_xyz(
            trimesh.points.PointCloud(mesh_2.vertices),
            write_colors=False,
            delimiter=" ")
        with open(xyz_path, "w") as f:
            f.write(xyz_data)
        # print(f"Processed label {label} successfully.")
        del mesh, mesh_2, xyz_data
        # skeleton = vol.skeleton.get(label)
        # with open(skeleton_path, "wb") as f:
        #     f.write(skeleton.to_swc())
        # def skeleton

    except Exception as e:
        print(f"Error processing label {label}: {e}")
        return None
    return label

def parallel_mesh_extraction():
    """
    Processes the segmentation labels in parallel, extracting and saving meshes.
    Tracks progress to ensure all meshes are processed.
    """
    tasks = []
    pool = multiprocessing.Pool(processes=num_workers)
    processed_labels = set()

    # Load existing processed labels if resuming
    if os.path.exists(processed_labels_path):
        with open(processed_labels_path, "rb") as f:
            processed_labels = pickle.load(f)

    expected_labels = set(all_labels)

    print("Extracting meshes for each label...")
    for label in tqdm(expected_labels):
        if label in processed_labels:
            # print(f"Label {label} already processed, skipping.")
            continue
        tasks.append(pool.apply_async(process_label, args=(label,)))

    pool.close()
    pool.join()

    processed_labels.update([task.get() for task in tasks if task.get() is not None])

    with open(processed_labels_path, "wb") as f:
        pickle.dump(processed_labels, f)

    missing_labels = expected_labels - processed_labels
    if missing_labels:
        print(f"Warning: Missing meshes detected! {len(missing_labels)} labels were not processed.")
        print("Please rerun the script to process missing meshes before finalizing.")
        return None

    return processed_labels

def finalize_mesh_extraction(processed_labels):
    """
    Finalizes the extraction process by saving the final list of processed labels.
    """
    with open(final_output_path, "wb") as f:
        pickle.dump(processed_labels, f)

    print(f"Final processed labels saved to {final_output_path}")


if __name__ == "__main__":
    # Step 1: Extract meshes in parallel and track progress
    processed_labels = parallel_mesh_extraction()

    # Step 2: Finalize if all meshes are processed
    if processed_labels:
        finalize_mesh_extraction(processed_labels)
    else:
        print("Finalization aborted due to missing meshes. Please rerun the script to complete processing.")

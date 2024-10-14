'''
helper functions to run Draco encoder
'''

import os

def draco_encoder_lauch(file_path, output_folder, draco_path="/Users/zyx/Desktop/Coding/draco/build", cl=7, opmode='os')-> int:
    '''
    Run Draco encoder for a file
    '''
    if opmode == 'os':
        output_path = run_draco(file_path, output_folder, draco_path, cl)
    elif opmode == 'subprocess':
        output_path = run_draco_subprocess(file_path, output_folder, draco_path, cl)
    else:
        raise ValueError(f"Invalid operation mode: {opmode}")
    print(f"Compressed file saved at {output_path}")
    return 0

def run_draco(file_path: str, output_folder: str, draco_path: str="/Users/zyx/Desktop/Coding/draco/build", cl: int =7)->str:
    file_name = os.path.basename(file_path)
    output_file_path = os.path.join(output_folder, os.path.splitext(os.path.basename(file_path))[0] + ".drc")
    command = f"{draco_path}/draco_encoder -cl {cl} -i {file_path} -o {output_file_path}"

    result = os.system(command)

    if result != 0:
        raise ValueError(f"Failed to run Draco encoder for {file_path}")
    return output_file_path
    
def run_draco_subprocess(file_path: str, output_folder: str, draco_path: str="/Users/zyx/Desktop/Coding/draco/build", cl: int =7):
    
    import subprocess

    file_name = os.path.basename(file_path)
    output_file_path = os.path.join(output_folder, os.path.splitext(file_name)[0] + ".drc")
    
    command = [
        os.path.join(draco_path, "draco_encoder"),
        "-cl", int(cl),
        "-i", file_path,
        "-o", output_file_path
    ]
    
    result = subprocess.run(command, text=True, capture_output=True)
    
    if result.returncode != 0:
        raise ValueError(f"Failed to run Draco encoder for {file_path}: {result.stderr}")
    
    return output_file_path


if __name__ == "__main__":

    file_path = "/path/to/a.ply"
    output_folder = "/path/to/output"
    draco_path = "/path/to/draco"
    cl = 7

    draco_encoder_lauch(file_path, output_folder, draco_path, cl)

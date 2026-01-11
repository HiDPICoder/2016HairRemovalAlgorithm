import subprocess
import os

def run_hair_removal(input_filename, output_dir=None, format=3, verbose=True, **kwargs):
    """
    Wrapper for hairrazor.exe using the subprocess module.
    
    Args:
        input_filename (str): Name of the PPM file (e.g., 'rr051107_CNP.ppm').
        output_dir (str): Path to the output directory.
        format (int): 0=original, 1=inverted, 2=both, 3=likeliest.
        verbose (bool): If True, enables verbose output (-v).
        **kwargs: Additional parameters for fine-tuning the algorithm.
    """
    
    # Path to the executable directory
    exe_dir = r"build\Release"
    exe_path = os.path.join(exe_dir, "hairrazor.exe")
    
    # Full path to the input file (located in the same folder as the exe)
    input_path = os.path.join(exe_dir, input_filename)
    
    if not os.path.exists(input_path):
        print(f"Error: Input file '{input_path}' not found.")
        return

    # Build the command using parameters defined in the source
    command = [exe_path, "-f", input_path, "-w", str(format)]
    
    if output_dir:
        command.extend(["-o", output_dir])
    
    if verbose:
        command.append("-v")
        
    # Argument mapping based on src/main.cpp
    arg_map = {
        'prune_percent': '-p',   # Default 0.05
        'min_prune': '-b',       # Default 3
        'max_prune': '-B',       # Default 40
        'dist_scaling': '-s',    # Default 0.2
        'min_dist': '-d',        # Default 20
        'max_dist': '-D',        # Default 30
        'junction_ratio': '-J',  # Default 0.1
        'skeleton_level': '-S',  # Default 2
        'morph_radius': '-r',    # Default 5
        'lambda_val': '-l',      # Default 0.2
    }

    for key, flag in arg_map.items():
        if key in kwargs:
            command.extend([flag, str(kwargs[key])])

    try:
        print(f"Executing: {' '.join(command)}")
        result = subprocess.run(command, capture_output=True, text=True, check=True)
        
        if result.stdout:
            print("Output:\n", result.stdout)
            
    except subprocess.CalledProcessError as e:
        print(f"Error (Exit Code {e.returncode}):\n", e.stderr)
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

# Example Usage
if __name__ == "__main__":
    # Just provide the filename since it's in \build\Release with the exe
    run_hair_removal(
        input_filename="rr051107_CNP.ppm", 
        output_dir="./results",
        format=3  # Likeliest mode
    )
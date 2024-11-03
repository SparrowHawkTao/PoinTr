import open3d as o3d
from pathlib import Path
import os
import numpy as np
import copy
import json
import random
import shutil

# Define constants for patch sizes
PATCH_SIZE_MAIN = 5120
PATCH_SIZE_OPPOSING = 5120
PATCH_SIZE_CROWN = 1568
PATCH_SIZE_MARGINLINE = 1024

# Define constants for data augmentation
ROTATION_ANGLE = np.pi/4  # 45 degrees
SCALE_FACTOR = 1.2
TRANSLATION_VECTOR = np.array([100, 100, 100])  # Move 10 units in x direction

def save_point_cloud(points, output_path, filename):
    """Utility function to save point cloud data"""
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    os.makedirs(output_path, exist_ok=True)
    o3d.io.write_point_cloud(os.path.join(output_path, filename), pcd)

def sample_and_save_point_clouds(path, ml_output_path=None, partial_output_path=None):
    """
    Reads point cloud files, samples them to predefined sizes, and saves:
    1. Individual point clouds (A, P, C, M) for ML input
    2. Combined partial point cloud (A+P+M) for visualization
    3. Ground truth point cloud (C+M) for evaluation
    
    Args:
        path: Input directory containing the .ply files
        ml_output_path: Output path for individual sampled clouds (for ML)
        partial_output_path: Output path for partial point cloud (A+P+M)
    """
    print("processing path: " + path)
    file_path = os.path.abspath(path)
    main, opposing, crown, marginline = None, None, None, None
    # read the point cloud files
    for j in os.listdir(file_path):
        if 'A.ply' in j:
           main = o3d.io.read_point_cloud(os.path.join(file_path, 'A.ply'))
           #o3d.visualization.draw_geometries([main])
        if 'P.ply' in j:
           opposing = o3d.io.read_point_cloud(os.path.join(file_path, 'P.ply'))
           #o3d.visualization.draw_geometries([opposing])
        if 'C.ply' in j:
           crown = o3d.io.read_point_cloud(os.path.join(file_path, 'C.ply'))
        if 'M.ply' in j:
           marginline = o3d.io.read_point_cloud(os.path.join(file_path, 'M.ply'))   
           #o3d.visualization.draw_geometries([marginline])
    # Check if all required point clouds are defined
    if main is None:
        raise ValueError("Main point cloud is not defined.")
    if opposing is None:
        raise ValueError("Opposing point cloud is not defined.")
    if crown is None:
        raise ValueError("Crown point cloud is not defined.")
    if marginline is None:
        raise ValueError("marginline point cloud is not defined.")
    # convert from point cloud to numpy 
    main,opposing,crown,marginline = convert_points(main,opposing,crown,marginline)
    #sample from main
    main_select = _get_random_chosen_points(main, PATCH_SIZE_MAIN)
    #sample from opposing
    opposing_select = _get_random_chosen_points(opposing, PATCH_SIZE_OPPOSING)
    #sample from crown
    crown_select = _get_random_chosen_points(crown, PATCH_SIZE_CROWN)
    #sample from marginline
    marginline_select = _get_random_chosen_points(marginline, PATCH_SIZE_MARGINLINE)

    # Save individual point clouds for ML if path provided
    if ml_output_path:
        save_point_cloud(main_select, ml_output_path, "A.ply")
        save_point_cloud(opposing_select, ml_output_path, "P.ply")
        save_point_cloud(crown_select, ml_output_path, "C.ply")
        save_point_cloud(marginline_select, ml_output_path, "M.ply")

    # Generate and save partial point cloud and ground truth if path provided
    if partial_output_path:
        # Combine A, P, and M points for partial
        partial_points = np.vstack((main_select, opposing_select, marginline_select))
        # Combine C and M points for ground truth
        ground_truth_points = np.vstack((crown_select, marginline_select))
        
        # Extract the case name from the path
        case_name = os.path.basename(path)
        
        # Save both files
        save_point_cloud(partial_points, partial_output_path, f"{case_name}_partial.ply")
        save_point_cloud(ground_truth_points, partial_output_path, f"{case_name}_ground_truth.ply")

    # Return the numpy arrays
    return main_select, opposing_select, crown_select, marginline_select

def _get_random_chosen_points(original, patch_size):
    original_idx = np.arange(len(original))
    try:
       selected_idx = np.random.choice(original_idx, size=patch_size, replace=False)
    except ValueError:
       selected_idx = np.random.choice(original_idx, size=patch_size, replace=True)   
    selected = np.zeros([patch_size, original.shape[1]], dtype='float32')
    selected[:] = original[selected_idx, :]
    return selected

def convert_points(main, opposing, crown, marginline):
    point_clouds = [main, opposing, crown, marginline]
    return [np.asarray(copy.deepcopy(pc).points) for pc in point_clouds]

def rotate_point_cloud(input_np, angle):
    rotation_matrix = np.array([[np.cos(angle), -np.sin(angle), 0],
                                [np.sin(angle), np.cos(angle), 0],
                                [0, 0, 1]])
    rotated_point_cloud = np.dot(input_np, rotation_matrix)
    return rotated_point_cloud

def scale_point_cloud(point_cloud, scale_factor):
    scaled_point_cloud = point_cloud * scale_factor
    return scaled_point_cloud

def translate_point_cloud(point_cloud, translation_vector):
    translated_point_cloud = point_cloud + translation_vector
    return translated_point_cloud

def rotate_point_clouds(main, opposing, crown, marginline, angle, output_path):
    """Rotates all point clouds by the specified angle around the Z-axis and saves to PLY files."""
    for data, suffix in [(main, 'A'), (opposing, 'P'), (crown, 'C'), (marginline, 'M')]:
        rotated = rotate_point_cloud(data.copy(), angle)
        save_point_cloud(rotated, output_path, f"{suffix}.ply")

def scale_point_clouds(main, opposing, crown, marginline, scale_factor, output_path):
    """Scales all point clouds by the specified factor and saves to PLY files."""
    for data, suffix in [(main, 'A'), (opposing, 'P'), (crown, 'C'), (marginline, 'M')]:
        scaled = scale_point_cloud(data.copy(), scale_factor)
        save_point_cloud(scaled, output_path, f"{suffix}.ply")

def translate_point_clouds(main, opposing, crown, marginline, translation_vector, output_path):
    """Translates all point clouds by the specified vector and saves to PLY files."""
    for data, suffix in [(main, 'A'), (opposing, 'P'), (crown, 'C'), (marginline, 'M')]:
        translated = translate_point_cloud(data.copy(), translation_vector)
        save_point_cloud(translated, output_path, f"{suffix}.ply")

def process_point_cloud_directory(input_path, ml_output_path, partial_output_path):
    """
    Processes point cloud datasets and creates both ML input files and partial visualization files.
    
    Args:
        input_path: Path to the root directory containing subdirectories with .ply files
        ml_output_path: Path where the ML input files will be saved
        partial_output_path: Path where the partial point clouds will be saved
    """
    folder_path = Path(input_path)
    idx = 0
    cleanup_output_directories(ml_output_path, partial_output_path)
    for item in folder_path.iterdir():
        if item.is_file():
            print(f'File: {item.name}, skip file.')
        elif item.is_dir():
            print(f'Processing directory: {item.name}')
            file_set = {'A.ply', 'P.ply', 'M.ply', 'C.ply'}
            for subitem in item.iterdir():
                if not subitem.name.endswith('.ply'):
                    continue
                if not subitem.name in file_set:
                    continue
                file_set.remove(subitem.name)
            if len(file_set) == 0:
                print(f'folder {item.name} index {idx} is complete.')
                idx = idx + 1
                
                # Create ML output directories
                fixed_size_path = os.path.join(ml_output_path, f"{item.name}_fixed_size")
                rotated_path = os.path.join(ml_output_path, f"{item.name}_rotated")
                scaled_path = os.path.join(ml_output_path, f"{item.name}_scaled")
                translated_path = os.path.join(ml_output_path, f"{item.name}_translated")
                
                # Create all directories
                for path in [fixed_size_path, rotated_path, scaled_path, translated_path]:
                    os.makedirs(path, exist_ok=True)
                
                # Sample and save point clouds with flat partial path
                main, opposing, crown, marginline = sample_and_save_point_clouds(
                    os.path.join(input_path, item.name),
                    fixed_size_path + "/",
                    partial_output_path  # Now passing the flat directory path
                )
                
                # Generate rotated versions
                rotate_point_clouds(main, opposing, crown, marginline, 
                                 ROTATION_ANGLE, rotated_path)
                
                # Generate scaled versions
                scale_point_clouds(main, opposing, crown, marginline, 
                                SCALE_FACTOR, scaled_path)
                
                # Generate translated versions
                translate_point_clouds(main, opposing, crown, marginline, 
                                    TRANSLATION_VECTOR, translated_path)
                
            else:
                print(f'folder {item.name} index {idx} is not complete')

def cleanup_output_directories(ml_output_path, partial_output_path):
    """
    Cleans up output directories before processing.
    
    Args:
        ml_output_path: Path for ML output files
        partial_output_path: Path for partial and ground truth files
    """
    # Remove directories if they exist
    if os.path.exists(ml_output_path):
        shutil.rmtree(ml_output_path)
    if os.path.exists(partial_output_path):
        shutil.rmtree(partial_output_path)
        
    # Create fresh directories
    os.makedirs(ml_output_path)
    os.makedirs(partial_output_path)
    print(f"Cleaned up and created fresh output directories")

def create_train_test_split(ml_output_path, train_ratio=0.8, seed=42):
    """
    Creates a JSON file containing train/test split information with all transformations.
    
    Args:
        ml_output_path: Path containing all transformed point cloud files
        train_ratio: Ratio of data to use for training (default: 0.8)
        seed: Random seed for reproducibility
    """
    # Set random seed for reproducibility
    random.seed(seed)
    
    # Define known transformations
    transformations = ['fixed_size', 'rotated', 'scaled', 'translated']
    
    # Get all directories
    all_dirs = [d for d in os.listdir(ml_output_path) 
                if os.path.isdir(os.path.join(ml_output_path, d))]
    
    # Extract unique case names by removing known transformation suffixes
    case_names = set()
    for dir_name in all_dirs:
        for transform in transformations:
            if dir_name.endswith(f"_{transform}"):
                case_name = dir_name[:-len(f"_{transform}")]
                case_names.add(case_name)
                break
    
    # Randomly split cases into train and test sets
    case_names = list(case_names)
    num_train = int(len(case_names) * train_ratio)
    random.shuffle(case_names)
    
    train_cases = case_names[:num_train]
    test_cases = case_names[num_train:]
    
    # Create lists with all transformations
    train_folders = [f"{case}_{transform}" 
                    for case in train_cases 
                    for transform in transformations]
    
    test_folders = [f"{case}_{transform}" 
                   for case in test_cases 
                   for transform in transformations]
    
    # Create split dictionary
    split_info = {
        'train': sorted(train_folders),
        'test': sorted(test_folders)
    }
    
    # Save to JSON file
    json_path = os.path.join(ml_output_path, 'train_test_split.json')
    with open(json_path, 'w') as f:
        json.dump(split_info, f, indent=4)
    
    print(f"Split complete: {len(train_folders)} training samples, {len(test_folders)} test samples")
    print(f"Split information saved to: {json_path}")
    return split_info

process_point_cloud_directory(
    '/Users/wutao/Desktop/ML_program/crown_data/0_crown_data_complete_original',
    '/Users/wutao/Desktop/ML_program/crown_data/1_crown_data_augmented',
    '/Users/wutao/Desktop/ML_program/crown_data/partial_crown_data'
)

create_train_test_split('/Users/wutao/Desktop/ML_program/crown_data/1_crown_data_augmented')
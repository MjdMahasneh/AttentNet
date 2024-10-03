size_L = (40, 40)
size_M = (25, 25)
size_S = (15, 15)

resized_voxel = (20, 20)
resize_spacing = [1, 1, 1]

config_LCL = {
    'resize_spacing': resize_spacing,
    'size_L': size_L,
    'size_M': size_M,
    'size_S': size_S,
    'resized_voxel': resized_voxel,

    'pos_path': 'S:/Deep_projects/FP_reduction_II/Data/processed_data/positive_samples',
    'neg_path': 'S:/Deep_projects/FP_reduction_II/Data/processed_data/negative_samples',

    'original_subset_directory_base': 'F:/Deep_projects/3D Object detection/Lung-Nodules/Tutorials/LUNA-Dataset/retuern to this-subset',
    'resampled_subset_directory_base': 'S:/Deep_projects/FP_reduction_II/Data/Resampled-1x1x1/subset',
    'candidates': 'F:/Deep_projects/3D Object detection/Lung-Nodules/Tutorials/LUNA-Dataset/candidates_V2.csv'
}

config_SCW = {
    'resize_spacing': resize_spacing,
    'size_L': size_L,
    'size_M': size_M,
    'size_S': size_S,
    'resized_voxel': resized_voxel,

    'pos_path': '/scratch/s.809508/FP_reduction/data/processed_data/positive_samples',
    'neg_path': '/scratch/s.809508/FP_reduction/data/processed_data/negative_samples',

    'original_subset_directory_base': '/scratch/s.809508/FP_reduction/data/raw/retuern_to_this-subset',
    'resampled_subset_directory_base': '/scratch/s.809508/FP_reduction/data/Resampled-1x1x1/subset',
    'candidates': '/scratch/s.809508/FP_reduction/data/candidates_V2.csv'
}

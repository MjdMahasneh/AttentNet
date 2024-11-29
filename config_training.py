
# Luna_raw: raw data folder downloaded from LUNA16 website
# Luna_segment: luna segmentation download from LUNA16 website
# Luna_data: temporary folder to store luna data
# Preprocess_result_path: final preprocessed data folder
# luna_abbr : a file that has the mapping between the Luna series IDs changed to their numerical ID (ranging from 0 to 887 for the Luna 888 candidate scans)
# luna_label : Luna annotations, with their series ID changed to thier numirical ID

config = {'luna_raw': 'F:/LUNA-Dataset/',
          'luna_segment': 'S:/seg-lungs-LUNA16/',

          'luna_data': 'S:/Luna_data/',
          'preprocess_result_path': 'S:/Preprocess_result_path/',

          'luna_abbr':  'S:/labels/shorter.csv',
          'luna_label': 'S:/labels/annos.csv',
          'preprocessing_backend': 'python'
          }

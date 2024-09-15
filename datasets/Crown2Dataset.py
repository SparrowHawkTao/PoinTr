import torch.utils.data as data
import numpy as np
import os, sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
import data_transforms
from .io import IO
import random
import os
import json
from .build import DATASETS
from utils.logger import *
import open3d as o3d

# References:
# - https://github.com/hzxie/GRNet/blob/master/utils/data_loaders.py

@DATASETS.register_module()
class crown2(data.Dataset):
    # def __init__(self, data_root, subset, class_choice = None):
    def __init__(self, config):
        self.sample_path = config.SAMPLE_PATH
        self.category_file = config.CATEGORY_FILE_PATH
        self.npoints = config.N_POINTS
        self.subset = config.subset

        # Load the dataset category: test and train
        with open(self.category_file) as f:
            self.dataset_categories = json.loads(f.read())

        self.sample_list = self._get_sample_list(self.subset)

    def _get_sample_list(self, subset):
        """Prepare file list for the dataset"""
        sample_list = []

        print_log('Collecting lines of subset: %s' % (subset), logger='CROWN2DATASET')
        lines = self.dataset_categories[subset]

        for line in lines:
            sample_list.append({
                'taxonomy_id': '02691156',
                'model_id': line,
                'path': self.sample_path % (line),
            })

        print_log('Complete collecting lines of the dataset. Total lines: %d' % len(sample_list), logger='CROWN2DATASET')
        return sample_list

    def convert_points(self,main,opposing,crown,implant):
   
        new_main = copy.deepcopy(main)
        new_opposing = copy.deepcopy(opposing)
        new_crown = copy.deepcopy(crown)
        new_implant = copy.deepcopy(implant)

        # scale values
        new_main_points = np.asarray(new_context.points)
        new_opposing_points = np.asarray(opposing.points)
        new_crown_points = np.asarray(shell.points)
        new_implant_points = np.asarray(marginline.points)
    
        return new_main_points, new_opposing_points, new_crown_points, new_implant_points

    def __getitem__(self, idx):
        sample = self.sample_list[idx]
        file_path = os.path.abspath(sample['path'])

        print_log('Get sample from path: %s' % file_path, logger='CROWN2DATASET')

        # T 1000 P & A 5120 C 1568

        main, opposing, crown, implant = None, None, None, None

        for j in os.listdir(file_path):
            if 'A.ply' in j:
               main = o3d.io.read_point_cloud(os.path.join(file_path, 'A.ply'))
               #o3d.visualization.draw_geometries([opposing])
            if 'P.ply' in j:
               opposing = o3d.io.read_point_cloud(os.path.join(file_path, 'P.ply'))
               #o3d.visualization.draw_geometries([master])
            if 'C.ply' in j:
               crown = o3d.io.read_point_cloud(os.path.join(file_path, 'C.ply'))
            if 'T.ply' in j:
               implant = o3d.io.read_point_cloud(os.path.join(file_path, 'T.ply'))   
               #o3d.visualization.draw_geometries([shell])

        # Check if all required point clouds are defined
        if main is None:
            raise ValueError("Main point cloud is not defined.")
        if opposing is None:
            raise ValueError("Opposing point cloud is not defined.")
        if crown is None:
            raise ValueError("Crown point cloud is not defined.")
        if implant is None:
            raise ValueError("Implant point cloud is not defined.")

        main,opposing,crown,implant = self.convert_points(main,opposing,crown,implant)

        #sample from main
        patch_size_main = 5120
        main_select = self._get_random_chosen_points(main, patch_size_main)

        #sample from opposing
        patch_size_opposing=5120
        opposing_select = self._get_random_chosen_points(opposing, patch_size_opposing)

        #sample from crown
        patch_size_crown=1568
        crown_select = self._get_random_chosen_points(crown, patch_size_crown)

        #sample from implant
        patch_size_implant=1000
        implant_select = self._get_random_chosen_points(implant, patch_size_implant)

        data_partial= torch.from_numpy(np.concatenate((main_select, implant_select, opposing_select), axis=0)).float()
        data_gt= torch.from_numpy(np.concatenate((crown_select, implant_select), axis=0)).float()

        return sample['taxonomy_id'], sample['model_id'], (data_partial, data_gt)

    def _get_random_chosen_points(self, original, patch_size):
        original_idx = np.arange(len(original))
        try:
           selected_idx = np.random.choice(original_idx, size=patch_size, replace=False)
        except ValueError:
           selected_idx = np.random.choice(original_idx, size=patch_size, replace=True)   
        selected = np.zeros([patch_size, original.shape[1]], dtype='float32')
        selected[:] = original[selected_idx, :]
        return selected

    def __len__(self):
        return len(self.sample_list)

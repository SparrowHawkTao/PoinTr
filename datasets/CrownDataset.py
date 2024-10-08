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


# References:
# - https://github.com/hzxie/GRNet/blob/master/utils/data_loaders.py

@DATASETS.register_module()
class crown(data.Dataset):
    # def __init__(self, data_root, subset, class_choice = None):
    def __init__(self, config):
        self.partial_points_path = config.PARTIAL_POINTS_PATH
        self.complete_points_path = config.COMPLETE_POINTS_PATH
        self.category_file = config.CATEGORY_FILE_PATH
        self.npoints = config.N_POINTS
        self.subset = config.subset

        # Load the dataset category: test and train
        with open(self.category_file) as f:
            self.dataset_categories = json.loads(f.read())

        self.file_list = self._get_file_list(self.subset)
        self.transforms = self._get_transforms(self.subset)

    def _get_transforms(self, subset):
        if subset == 'train':
            return data_transforms.Compose([{
                'callback': 'RandomSamplePoints',
                'parameters': {
                    'n_points': 14400
                },
                'objects': ['partial']
            }, {
                'callback': 'RandomSamplePoints',
                'parameters': {
                    'n_points': self.npoints
                },
                'objects': ['gt']
            },{
                'callback': 'ToTensor',
                'objects': ['partial', 'gt']
            }])
        else:
            return data_transforms.Compose([{
                'callback': 'RandomSamplePoints',
                'parameters': {
                    'n_points': 14400
                },
                'objects': ['partial']
            },
            {
                'callback': 'RandomSamplePoints',
                'parameters': {
                    'n_points': self.npoints
                },
                'objects': ['gt']
            },
            {
                'callback': 'ToTensor',
                'objects': ['partial', 'gt']
            }])

    def _get_file_list(self, subset):
        """Prepare file list for the dataset"""
        file_list = []

        print_log('Collecting files of subset: %s' % (subset), logger='CROWNDATASET')
        samples = self.dataset_categories[subset]

        for s in samples:
            file_list.append({
                'taxonomy_id': '02691156',
                'model_id': s,
                'partial_path': self.partial_points_path % (s),
                'gt_path': self.complete_points_path % (s),
            })

        print_log('Complete collecting files of the dataset. Total files: %d' % len(file_list), logger='CROWNDATASET')
        return file_list

    def __getitem__(self, idx):
        sample = self.file_list[idx]
        data = {}

        for ri in ['partial', 'gt']:
            file_path = sample['%s_path' % ri]
            data[ri] = IO.get(file_path).astype(np.float32)
        # assert data['gt'].shape[0] == self.npoints

        if self.transforms is not None:
            data = self.transforms(data)
        return sample['taxonomy_id'], sample['model_id'], (data['partial'], data['gt'])

    def __len__(self):
        return len(self.file_list)

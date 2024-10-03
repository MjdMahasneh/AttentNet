import numpy as np
import torch
from torch.utils.data import Dataset
import os
import time
from data_helpers import load_itk_image, normalizePlanes, get_patch, zoom_patch, get_img_list, crop_patch
from scipy import ndimage
import matplotlib.pyplot as plt
from visualization_helpers import visualize_multiple_3D_images


class LUNA16Dataset(Dataset):
    """LINA16 False positive reduction dataset."""

    def __init__(self, pos_path=None, neg_path=None, outer_sets=[], size_L=None, size_M=None, size_S=None,
                 resized_voxel=None, num_sets=10, phase='train'):

        assert phase == 'train' or phase == 'test'
        assert pos_path != None and neg_path != None
        assert size_L != None and size_M != None, size_S != None
        assert resized_voxel != None
        for i in outer_sets:
            assert int(i) in range(num_sets)

        print('init dataset ...')

        seed = 999
        np.random.seed(seed)

        self.pos_path = pos_path
        self.neg_path = neg_path
        self.size_L, self.size_M, self.size_S = size_L, size_M, size_S

        self.resized_voxel = resized_voxel

        self.pos_img_list = []
        self.neg_img_list = []
        if phase != 'test':
            for i in range(num_sets):
                if str(i) not in outer_sets:
                    pos_subset = get_img_list(self.pos_path + '/subset' + str(i))
                    self.pos_img_list.extend(pos_subset)

                    neg_subset = get_img_list(self.neg_path + '/subset' + str(i))
                    self.neg_img_list.extend(neg_subset)
        else:
            for i in range(num_sets):
                if str(i) in outer_sets:
                    pos_subset = get_img_list(self.pos_path + '/subset' + str(i))
                    self.pos_img_list.extend(pos_subset)

                    neg_subset = get_img_list(self.neg_path + '/subset' + str(i))
                    self.neg_img_list.extend(neg_subset)

        self.dataset_len = len(self.pos_img_list)

        unified_seed = 999
        np.random.seed(unified_seed)
        np.random.shuffle(self.pos_img_list)
        np.random.shuffle(self.neg_img_list)

        print('number of positive and negative samples ({}) : '.format(phase), self.dataset_len, len(self.neg_img_list))

    def __len__(self):
        return self.dataset_len

    def __getitem__(self, idx):

        if idx == 0:
            t = time.time()
            np.random.seed(int(str(t % 1)[2:7]))
            np.random.shuffle(self.neg_img_list)

        if idx >= len(self.neg_img_list):
            print(
                'idx is larger than len(self.neg_img_list). This should never happen. idx {}, len(self.neg_img_list) {}'.format(
                    idx, len(self.neg_img_list)))
            idx = idx % len(self.neg_img_list)

        pos_sample = np.load(self.pos_img_list[idx])

        pos_sample_L = crop_patch(pos_sample, self.size_L[0], self.size_L[1])
        pos_sample_M = crop_patch(pos_sample, self.size_M[0], self.size_M[1])
        pos_sample_S = crop_patch(pos_sample, self.size_S[0], self.size_S[1])

        pos_sample_L = zoom_patch(pos_sample_L, self.resized_voxel[0], self.resized_voxel[1])
        pos_sample_M = zoom_patch(pos_sample_M, self.resized_voxel[0], self.resized_voxel[1])
        pos_sample_S = zoom_patch(pos_sample_S, self.resized_voxel[0], self.resized_voxel[1])

        neg_sample = np.load(self.neg_img_list[idx])

        neg_sample_L = crop_patch(neg_sample, self.size_L[0], self.size_L[1])
        neg_sample_M = crop_patch(neg_sample, self.size_M[0], self.size_M[1])
        neg_sample_S = crop_patch(neg_sample, self.size_S[0], self.size_S[1])

        neg_sample_L = zoom_patch(neg_sample_L, self.resized_voxel[0], self.resized_voxel[1])
        neg_sample_M = zoom_patch(neg_sample_M, self.resized_voxel[0], self.resized_voxel[1])
        neg_sample_S = zoom_patch(neg_sample_S, self.resized_voxel[0], self.resized_voxel[1])

        target_pos = torch.ones(1)
        target_neg = torch.zeros(1)

        return torch.from_numpy(np.array(pos_sample_L)).unsqueeze(0).float(), torch.from_numpy(
            np.array(pos_sample_M)).unsqueeze(0).float(), torch.from_numpy(np.array(pos_sample_S)).unsqueeze(0).float(), \
            torch.from_numpy(neg_sample_L).unsqueeze(0).float(), torch.from_numpy(neg_sample_M).unsqueeze(
            0).float(), torch.from_numpy(neg_sample_S).unsqueeze(0).float(), \
            target_pos.float(), target_neg.float()


class TestingLUNA16Dataset(Dataset):
    """LINA16 False positive reduction testing dataset."""

    def __init__(self, original_subset_directory_base=None, resampled_subset_directory_base=None,
                 resize_spacing=None,
                 size_L=None, size_M=None, size_S=None,
                 resized_voxel=None,
                 candidates_list=None, outer_set=None):

        print('init dataset ...')

        assert original_subset_directory_base != None and resampled_subset_directory_base != None
        assert resize_spacing != None
        assert candidates_list != None
        assert outer_set != None
        assert size_L != None and size_M != None, size_S != None
        assert resized_voxel != None

        self.original_subset_directory_base = original_subset_directory_base
        self.resampled_subset_directory_base = resampled_subset_directory_base
        self.resize_spacing = resize_spacing
        self.size_L, self.size_M, self.size_S = size_L, size_M, size_S
        self.resized_voxel = resized_voxel
        self.candidates_list = candidates_list
        self.outer_set = outer_set

        subset_dir = self.original_subset_directory_base + str(self.outer_set)
        flist = os.listdir(subset_dir)

        self.subset_outer_list = []
        for file in flist:
            if file.endswith(".mhd"):
                file = file[:-4]
                self.subset_outer_list.append(file)

        print('subset {} contains {} scans : '.format(self.outer_set, len(self.subset_outer_list)))

    def __len__(self):
        return len(self.subset_outer_list)

    def __getitem__(self, idx):

        file = self.subset_outer_list[idx]

        test_list = []
        test_patch_list_L = []
        test_patch_list_M = []
        test_patch_list_S = []
        target_list = []

        for cand in self.candidates_list:
            if (cand[0] == file):
                test_list.append(cand)

        original_file_path = self.original_subset_directory_base + str(self.outer_set) + '/' + file + '.mhd'
        new_file_path = self.resampled_subset_directory_base + str(self.outer_set) + '/' + file + '.npy'

        volume_image, numpy_origin, numpy_spacing = load_itk_image(original_file_path)
        new_volume = np.load(new_file_path)

        new_volume = normalizePlanes(new_volume)

        for candidate_item in test_list:
            patch_L = get_patch(new_volume, candidate_item, numpy_origin, self.resize_spacing, self.size_L[0],
                                self.size_L[1])
            patch_M = crop_patch(patch_L, self.size_M[0], self.size_M[1])
            patch_S = crop_patch(patch_L, self.size_S[0], self.size_S[1])

            patch_L = zoom_patch(patch_L, self.resized_voxel[0], self.resized_voxel[1])
            patch_M = zoom_patch(patch_M, self.resized_voxel[0], self.resized_voxel[1])
            patch_S = zoom_patch(patch_S, self.resized_voxel[0], self.resized_voxel[1])

            test_patch_list_L.append(patch_L)
            test_patch_list_M.append(patch_M)
            test_patch_list_S.append(patch_S)

            target_list.append(candidate_item)

        return torch.from_numpy(np.array(test_patch_list_L)).unsqueeze(1).float(), \
            torch.from_numpy(np.array(test_patch_list_M)).unsqueeze(1).float(), \
            torch.from_numpy(np.array(test_patch_list_S)).unsqueeze(1).float(), \
            target_list


if __name__ == '__main__':

    from torch.utils.data import DataLoader


    def shuffle_in_unison(a, b, c, d):
        assert len(a) == len(b) == len(c) == len(d)

        idxs = np.arange(len(a))
        np.random.shuffle(idxs)

        return a[idxs], b[idxs], c[idxs], d[idxs]


    pos_path = 'S:/Deep_projects/FP_reduction_II/Data/processed_data_Approach-2_prime/positive_samples'
    neg_path = 'S:/Deep_projects/FP_reduction_II/Data/processed_data_Approach-2_prime/negative_samples'
    outer_sets = ['0']
    num_sets = 10

    dataset = LUNA16Dataset(pos_path=pos_path, neg_path=neg_path, size_L=(26, 40, 40), size_M=(10, 30, 30),
                            size_S=(6, 20, 20), resized_voxel=(6, 20, 20), outer_sets=outer_sets, num_sets=num_sets,
                            phase='train')
    data_loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=1, pin_memory=False)

    tic = time.time()
    for i, (dicts_axi) in enumerate(data_loader):
        pos_sample_L, pos_sample_M, pos_sample_S, neg_sample_L, neg_sample_M, neg_sample_S, Y_pos, Y_neg = dicts_axi
        batch_L = torch.cat([pos_sample_L, neg_sample_L], dim=0)
        batch_M = torch.cat([pos_sample_M, neg_sample_M], dim=0)
        batch_S = torch.cat([pos_sample_S, neg_sample_S], dim=0)

        print('batch_L.size(), batch_M.size(), batch_S.size()', batch_L.size(), batch_M.size(), batch_S.size())

        target = torch.cat([Y_pos, Y_neg], dim=0)

        batch_L, batch_M, batch_S, target = shuffle_in_unison(batch_L, batch_M, batch_S, target)

        raise Exception('stop!')
    toc = time.time()
    print('elapsed {}'.format(toc - tic))

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from logging import getLogger

import numpy as np
from torch.utils.data import Dataset
import torch
#import kornia.augmentation as K
#import torch.nn as nn

logger = getLogger()


def randomCrop(img, mask, width, height):
    assert img.shape[1] == mask.shape[0]
    assert img.shape[2] == mask.shape[1]

    x = np.random.randint(0, img.shape[1] - width)
    y = np.random.randint(0, img.shape[2] - height)
    img = img[:,x:x+width,y:y+height]
    mask = mask[x:x+width,y:y+height]
    return img, mask


class DataloaderEval(Dataset):
    def __init__(self, 
                 img, 
                 coords, 
                 psize = 128,
                 labels=None,
                 lab_ind=True):

        super(DataloaderEval, self).__init__()
        self.img = img
        self.coord = coords
        self.lab_ind = lab_ind
        self.labels = labels
        self.psize = psize
        
        
    def __len__(self):
        return self.coord.shape[0]
        
    def __getitem__(self, idx):
        
        image = self.img[:,self.coord[idx,0]-self.psize//2:self.coord[idx,0]+self.psize//2+self.psize%2,
                              self.coord[idx,1]-self.psize//2:self.coord[idx,1]+self.psize//2+self.psize%2]

        image = torch.from_numpy(image.astype(np.float32))/255
        
        if self.lab_ind:
            mask_img = self.labels[self.coord[idx,0]-self.psize//2:self.coord[idx,0]+self.psize//2+self.psize%2,
                                  self.coord[idx,1]-self.psize//2:self.coord[idx,1]+self.psize//2+self.psize%2]
            mask_img = torch.from_numpy(mask_img.astype(np.float32))
            return image, mask_img
        
        return image


class DataloaderLSTM(Dataset):
    def __init__(self, 
                 img, 
                 coords, 
                 lstm_length = 6,
                 samples=None,
                 labels=None,
                 lab_ind=True):

        super(DataloaderLSTM, self).__init__()
        self.img = img
        self.coord = coords
        self.samples = samples
        self.lab_ind = lab_ind
        self.labels = labels
        self.lstm_length = lstm_length
        self.temp_size = self.img.shape[0]
        
        
    def __len__(self):
        if self.samples:
            return self.samples
        else:
            return self.coord.shape[0]
        
    def __getitem__(self, idx):
        if self.lab_ind:
            tmp_idx = np.random.randint(self.lstm_length,self.temp_size-1)
            image = self.img[tmp_idx-self.lstm_length:tmp_idx,:,self.coord[idx,0],self.coord[idx,1]]
            
        if not self.lab_ind:
            image = self.img[-self.lstm_length:,:,self.coord[idx,0],self.coord[idx,1]]

        image = torch.from_numpy(image.astype(np.float32))
        
        if self.lab_ind:
            lab = np.sum(self.labels[tmp_idx+1,self.coord[idx,0],self.coord[idx,1]])
            lab = torch.from_numpy(np.array(lab).astype(np.float32))
            return image, lab
        
        return image
    
    
class MemmapNPYCropDataset(Dataset):
    def __init__(self, 
                 file_img, 
                 file_lab, 
                 coords, 
                 max_time,
                 psize,
                 tile_shape,
                 features,
                 samples):

        super(MemmapNPYCropDataset, self).__init__()
        self.file_img = file_img
        self.file_lab = file_lab
        self.coord = coords
        self.samples = samples
        self.psize = psize
        self.max_time = max_time
        self.tile_shape = tile_shape
        self.features = features
        
        
    def __len__(self):
        return self.samples
        
    def __getitem__(self, idx):
        file_indx = np.random.randint(0,len(self.file_img))
        time_indx = np.random.randint(0,self.max_time)
    
        
        # Get the filename and corresponding coordinate
        filename = self.file_img[file_indx]
        refname = self.file_lab[file_indx]
        coords = self.coord[file_indx][time_indx]
        if coords.shape[0] <= idx:
            idx = np.random.randint(0,len(self.file_img)) 

        row, col = coords[idx]
        # Define the region to load
        row_start = max(0, row - self.psize // 2)
        col_start = max(0, col - self.psize // 2)
        row_end = row_start + self.psize +self.psize%2
        col_end = col_start + self.psize + +self.psize%2

        # Open the file as a memory-mapped array
        img = np.memmap(filename, dtype=np.float16, mode='r', shape=(self.features, self.max_time, self.tile_shape, self.tile_shape))
        ref = np.memmap(refname, dtype=np.uint8, mode='r', shape=(self.max_time, self.tile_shape, self.tile_shape))
        crop_img = torch.from_numpy(img[:, time_indx, row_start:row_end, col_start:col_end].copy().astype(np.float32))
        crop_ref = torch.from_numpy(ref[time_indx, row_start:row_end, col_start:col_end].copy().astype(np.float32))

        return crop_img, crop_ref
    

class NumpyTileDataset(Dataset):
    def __init__(self, file_img, file_lab, coords, psize, samples=None, max_time=1):
        self.file_img = file_img
        self.file_lab = file_lab
        self.coords = coords
        self.tile_size = psize + (psize % 2)  # ensure even size
        self.samples = samples
        self.max_time = max_time

    def __len__(self):
        return self.samples or len(self.coords[0])

    def __getitem__(self, idx):
        file_idx = np.random.randint(len(self.file_img))
        time_idx = np.random.randint(self.max_time)
        
        print(file_idx)
        print(time_idx)

        coord_list = self.coords[file_idx][time_idx]
        if idx >= len(coord_list):
            idx = np.random.randint(len(coord_list))

        x, y = coord_list[idx]

        data = np.load(self.file_img[file_idx], mmap_mode='r')
        label = np.load(self.file_lab[file_idx], mmap_mode='r')

        half_size = self.tile_size // 2
        row_start = max(0, x - half_size)
        col_start = max(0, y - half_size)
        row_end = row_start + self.tile_size
        col_end = col_start + self.tile_size

        tile = data[:, time_idx, row_start:row_end, col_start:col_end]
        gt = label[time_idx, row_start:row_end, col_start:col_end]

        tile = torch.from_numpy(tile.astype(np.float32)) / 255.0
        gt = torch.from_numpy(gt.astype(np.float32))

        return tile, gt

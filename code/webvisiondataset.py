from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import random
import numpy as np
from PIL import Image
import torch
import os
import copy


class webvision_dataset(Dataset):
    def __init__(self, root, mode ='train', num_classes=50, transform = None,target_transform = None): 

        self.root = root
        self.transform = transform
        self.target_transform = target_transform
        self.mode = mode

        if self.mode == 'test':
            with open(os.path.join(self.root, 'info/val_filelist.txt')) as f:
                lines = f.readlines()

            self.data = []
            self.targets = []
            for line in lines:
                img, target = line.split()
                target = int(target)
                if target < num_class:
                    self.data.append(img)
                    self.targets.append(target)

        else:
            with open(self.root+'info/train_filelist_google.txt') as f:
                lines=f.readlines()

            data = []
            self.targets = []
            for line in lines:
                img, target = line.split()
                target = int(target)
                if target < num_class:
                    data.append(img)
                    self.targets.append(target)
            
            if self.mode == 'train':
                self.data = data
        self.copy_data()

    def copy_data(self):
        #pass
        self.whole_data = self.data.copy()
        self.whole_targets = copy.deepcopy(self.targets)

    def switch_data(self): 
        #pass
        self.data = self.whole_data
        self.targets = self. whole_targets

    def ajust_base_indx_temp(self, idx):
        #pass
        new_data = self.whole_data[idx,...]
        target_np = np.array(self.whole_targets)
        new_targets = target_np[idx].tolist()

        self.data = new_data
        self.targets = new_targets


    def __getitem__(self, index):
        if self.mode == 'train':
            img_path = self.data[index]
            target = self.targets[index]
            image = Image.open(os.path.join(self.root, img_path)).convert('RGB')
            img = self.transform(image)

            return img, target, index
        
        elif self.mode=='test':
            img_path = self.data[index]
            target = self.targets[index]    
            image = Image.open(self.root+'val_images_256/'+img_path).convert('RGB')   
            img = self.transform(image) 
            return img, target

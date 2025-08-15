import math
import os
import random
from copy import deepcopy
from PIL import Image
from scipy.ndimage.interpolation import zoom
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from util.my_transform import random_rot_flip, random_rotate, blur, obtain_cutmix_box


class RGBDataset(Dataset):
    def __init__(self, name, root, mode, size=None, id_path=None, nsample=None):
        self.name = name
        self.root = root
        self.mode = mode
        self.size = size

        if mode == 'train_l' or mode == 'train_u':
            with open(id_path, 'r') as f:
                self.ids = f.read().splitlines()
            if mode == 'train_l' and nsample is not None:
                self.ids *= math.ceil(nsample / len(self.ids))
                self.ids = self.ids[:nsample]
        else:
            with open('splits/%s/valtest.txt' % name, 'r') as f:
                self.ids = f.read().splitlines()

    def __getitem__(self, item):
        id = self.ids[item]
        #*****Main modification range, starting from here*****
        img = Image.open(os.path.join(self.root, id)).convert('L') # uint8
        img = np.array(img)
        img = img.astype(np.float32) / 255.0
        # Replace 'image' in id with 'label'
        id_mask = id
        id_mask = id_mask.replace('image', 'label')
        id_mask = id_mask.replace('jpg', 'png')
        mask = Image.open(os.path.join(self.root, id_mask)).convert('L')
        mask = np.array(mask)
        mask[mask == 255] = 1
        if self.mode == 'val':
            # Missing channel dimension in previous processing, adding a new channel dimension
            # This is a special handling, only adding channel for validation set, as explained in the main code
            img = np.expand_dims(img, axis=0)
            mask = np.expand_dims(mask, axis=0)
            return torch.from_numpy(img).float(), torch.from_numpy(mask).long()
        # *****Main modification range, ending here*****

        if random.random() > 0.5:
            img, mask = random_rot_flip(img, mask)
        elif random.random() > 0.5:
            img, mask = random_rotate(img, mask)
        x, y = img.shape
        img = zoom(img, (self.size / x, self.size / y), order=0)
        mask = zoom(mask, (self.size / x, self.size / y), order=0)

        if self.mode == 'train_l':
            return torch.from_numpy(img).unsqueeze(0).float(), torch.from_numpy(np.array(mask)).long()

        img = Image.fromarray((img * 255).astype(np.uint8))
        img_s1, img_s2 = deepcopy(img), deepcopy(img)
        img = torch.from_numpy(np.array(img)).unsqueeze(0).float() / 255.0

        if random.random() < 0.8:
            img_s1 = transforms.ColorJitter(0.5, 0.5, 0.5, 0.25)(img_s1)
        img_s1 = blur(img_s1, p=0.5)
        cutmix_box1 = obtain_cutmix_box(self.size, p=0.5)
        img_s1 = torch.from_numpy(np.array(img_s1)).unsqueeze(0).float() / 255.0

        if random.random() < 0.8:
            img_s2 = transforms.ColorJitter(0.5, 0.5, 0.5, 0.25)(img_s2)
        img_s2 = blur(img_s2, p=0.5)
        cutmix_box2 = obtain_cutmix_box(self.size, p=0.5)
        img_s2 = torch.from_numpy(np.array(img_s2)).unsqueeze(0).float() / 255.0
        return img, img_s1, img_s2, cutmix_box1, cutmix_box2
    def __len__(self):
        return len(self.ids)

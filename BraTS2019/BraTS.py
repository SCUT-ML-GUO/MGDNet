import os
import torch
from torch.utils.data import Dataset
import random
import numpy as np
from torchvision.transforms import transforms
import h5py
import torch.nn.functional as F
from scipy.ndimage import map_coordinates, gaussian_filter, zoom
from scipy import interpolate
import numbers

class RandomRotate3D:
    def __init__(self, angle_spectrum=15, p=0.5):
        self.angle_spectrum = angle_spectrum
        self.p = p

    def __call__(self, sample):
        image, mask = sample['image'], sample['label']
        
        if random.random() < self.p:
            angle_x = np.random.uniform(-self.angle_spectrum, self.angle_spectrum)
            angle_y = np.random.uniform(-self.angle_spectrum, self.angle_spectrum)
            angle_z = np.random.uniform(-self.angle_spectrum, self.angle_spectrum)
            
            image = self.rotate_3d(image, angle_x, angle_y, angle_z, order=1) 
            mask = self.rotate_3d(mask, angle_x, angle_y, angle_z, order=0)   
            
        return {'image': image, 'label': mask}

    def rotate_3d(self, volume, angle_x, angle_y, angle_z, order=1):
        angle_x = np.deg2rad(angle_x)
        angle_y = np.deg2rad(angle_y)
        angle_z = np.deg2rad(angle_z)

        if volume.ndim == 3: 
            depth, height, width = volume.shape
            channels = 1
        else: 
            channels, depth, height, width = volume.shape
        
        z, y, x = np.meshgrid(np.arange(depth), 
                             np.arange(height), 
                             np.arange(width), 
                             indexing='ij')
        
        center = np.array([(depth-1)/2, (height-1)/2, (width-1)/2])

        coords = np.stack([x, y, z], axis=0).astype(np.float32) 
        for i in range(3):
            coords[i] -= center[i]
        
        coords_flat = coords.reshape(3, -1) 
        
        Rx = np.array([[1, 0, 0],
                      [0, np.cos(angle_x), -np.sin(angle_x)],
                      [0, np.sin(angle_x), np.cos(angle_x)]])
        
        Ry = np.array([[np.cos(angle_y), 0, np.sin(angle_y)],
                      [0, 1, 0],
                      [-np.sin(angle_y), 0, np.cos(angle_y)]])
        
        Rz = np.array([[np.cos(angle_z), -np.sin(angle_z), 0],
                      [np.sin(angle_z), np.cos(angle_z), 0],
                      [0, 0, 1]])
        
        R = Rz @ Ry @ Rx
        
        coords_rotated = R @ coords_flat  
        
        for i in range(3):
            coords_rotated[i] += center[i]
        
        coords_rotated = coords_rotated.reshape(3, depth, height, width)
        
        if volume.ndim == 4: 
            rotated_volume = np.zeros_like(volume)
            for c in range(channels):
                rotated_volume[c] = map_coordinates(volume[c], coords_rotated, 
                                                  order=order, mode='constant', cval=0)
        else:
            rotated_volume = map_coordinates(volume, coords_rotated, 
                                           order=order, mode='constant', cval=0)
        
        return rotated_volume

class RandomElasticDeformation3D:
    def __init__(self, alpha=50, sigma=10, p=0.3):
        self.alpha = alpha
        self.sigma = sigma
        self.p = p

    def __call__(self, sample):
        image, mask = sample['image'], sample['label']
        
        if random.random() < self.p:
            image, mask = self.elastic_deform_3d(image, mask)
            
        return {'image': image, 'label': mask}

    def elastic_deform_3d(self, image, mask):
        if image.ndim == 4:  # [C, H, W, D]
            channels, height, width, depth = image.shape
            shape = (height, width, depth)
        else:
            height, width, depth = image.shape
            shape = (height, width, depth)
            channels = 1
        
        dx = gaussian_filter((np.random.rand(*shape) * 2 - 1), self.sigma, mode="constant", cval=0) * self.alpha
        dy = gaussian_filter((np.random.rand(*shape) * 2 - 1), self.sigma, mode="constant", cval=0) * self.alpha
        dz = gaussian_filter((np.random.rand(*shape) * 2 - 1), self.sigma, mode="constant", cval=0) * self.alpha

        y, x, z = np.meshgrid(np.arange(height), np.arange(width), np.arange(depth), indexing='ij')
        
        indices = [y + dy, x + dx, z + dz]

        if image.ndim == 4:
            deformed_image = np.zeros_like(image)
            for c in range(channels):
                deformed_image[c] = map_coordinates(image[c], indices, order=1, mode='constant').reshape(shape)
        else:
            deformed_image = map_coordinates(image, indices, order=1, mode='constant').reshape(shape)
        
        deformed_mask = map_coordinates(mask, indices, order=0, mode='constant').reshape(shape)
        
        return deformed_image, deformed_mask

class GammaCorrection:
    """伽马校正"""
    def __init__(self, gamma_range=(0.7, 1.5), p=0.5):
        self.gamma_range = gamma_range
        self.p = p

    def __call__(self, sample):
        image, mask = sample['image'], sample['label']
        
        if random.random() < self.p:
            gamma = np.random.uniform(self.gamma_range[0], self.gamma_range[1])
            image = self.apply_gamma(image, gamma)
            
        return {'image': image, 'label': mask}

    def apply_gamma(self, image, gamma):
        min_val = image.min()
        max_val = image.max()
        
        if max_val > min_val:
            image_normalized = (image - min_val) / (max_val - min_val)
            image_gamma = np.power(image_normalized, gamma)
            image = image_gamma * (max_val - min_val) + min_val
        
        return image

class RandomFlip3D:
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, sample):
        image, mask = sample['image'], sample['label']
        
        axes = [0, 1, 2] 
        
        for axis in axes:
            if random.random() < self.p:
                image = np.flip(image, axis=axis + 1).copy() 
                mask = np.flip(mask, axis=axis).copy()
                
        return {'image': image, 'label': mask}

class ToTensor3D:
    def __call__(self, sample):
        image, mask = sample['image'], sample['label']
        
        image = torch.from_numpy(image.astype(np.float32))
        mask = torch.from_numpy(mask.astype(np.int64))
        
        return {'image': image, 'label': mask}

class RandomCrop(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        init_image, init_label = sample['image'], sample['label']

        (c, w, h, d) = init_image.shape

        w1 = np.random.randint(0, w - self.output_size[0])
        h1 = np.random.randint(0, h - self.output_size[1])
        d1 = np.random.randint(0, d - self.output_size[2])

        label = init_label[w1:w1 + self.output_size[0], h1:h1 + self.output_size[1], d1:d1 + self.output_size[2]]
        image = init_image[:,w1:w1 + self.output_size[0], h1:h1 + self.output_size[1], d1:d1 + self.output_size[2]]
        cnt =  np.count_nonzero(label)
        for i in range(3):
            w1 = np.random.randint(0, w - self.output_size[0])
            h1 = np.random.randint(0, h - self.output_size[1])
            d1 = np.random.randint(0, d - self.output_size[2])
            sum = np.count_nonzero(init_label[w1:w1 + self.output_size[0], h1:h1 + self.output_size[1], d1:d1 + self.output_size[2]])
            if sum > cnt:
                label = init_label[w1:w1 + self.output_size[0], h1:h1 + self.output_size[1], d1:d1 + self.output_size[2]]
                image = init_image[:, w1:w1 + self.output_size[0], h1:h1 + self.output_size[1], d1:d1 + self.output_size[2]]
                cnt = sum
        return {'image': image, 'label': label}

class CenterCrop(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample['image'], sample['label']

        (c,w, h, d) = image.shape

        w1 = int(round((w - self.output_size[0]) / 2.))
        h1 = int(round((h - self.output_size[1]) / 2.))
        d1 = int(round((d - self.output_size[2]) / 2.))

        label = label[w1:w1 + self.output_size[0], h1:h1 + self.output_size[1], d1:d1 + self.output_size[2]]
        image = image[:,w1:w1 + self.output_size[0], h1:h1 + self.output_size[1], d1:d1 + self.output_size[2]]

        return {'image': image, 'label': label}
        
class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""
    def __call__(self, sample):
        image = sample['image']
        label = sample['label']

        image = torch.from_numpy(image).float()
        label = torch.from_numpy(label).long()

        return {'image': image, 'label': label}

class BraTS(Dataset):
    def __init__(self,data_path, file_path,transform=None):
        with open(file_path, 'r') as f:
            self.paths = [os.path.join(data_path, x.strip().split('/')[-1]) for x in f.readlines()]
        self.transform = transform

    def __getitem__(self, item):
        h5f = h5py.File(self.paths[item], 'r')
        image = h5f['image'][:]
        label = h5f['label'][:]
        label[label == 4] = 3
        sample = {'image': image, 'label': label}
        if self.transform:
            sample = self.transform(sample)
        return sample['image'], sample['label']

    def __len__(self):
        return len(self.paths)

    def collate(self, batch):
        return [torch.cat(v) for v in zip(*batch)]


if __name__ == '__main__':
    from torchvision import transforms
    data_path = "/dataset/data"
    test_txt = "/postgraduate/test.txt"
    test_set = BraTS(data_path,test_txt,transform=transforms.Compose([
        RandomRotFlip(),
        RandomCrop((160,160,128)),
        GaussianNoise(p=0.1),
        ToTensor()
    ]))
    d1 = test_set[0]
    image,label = d1
    print(image.shape)
    print(label.shape)
    print(np.unique(label))

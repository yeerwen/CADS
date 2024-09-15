import numpy as np
import random
import torch
import matplotlib.pyplot as plt
import nibabel as nib
import cv2
import os
import math
from torch.utils.data import Dataset
from batchgenerators.transforms import Compose
from batchgenerators.transforms.color_transforms import BrightnessMultiplicativeTransform, GammaTransform, \
    BrightnessTransform, ContrastAugmentationTransform
from batchgenerators.transforms.noise_transforms import GaussianNoiseTransform, GaussianBlurTransform
from einops.einops import rearrange

class RandomMaskingGenerator:
    def __init__(self, input_size=(24//8, 224//16, 224//16), mask_ratio=0.75, regular=True):
        if not isinstance(input_size, tuple):
            input_size = (input_size,) * 2

        self.depth, self.height, self.width = input_size
        self.num_patches = self.height * self.width * self.depth
        self.num_mask = int(mask_ratio * self.num_patches)
        self.regular = regular

        if regular:
            candidate_list = []
            while True:  # add more
                for j in range(4):
                    candidate = np.zeros(4)
                    candidate[j] = 1
                    candidate_list.append(candidate)
                if len(candidate_list) == self.num_patches // 4:
                    break
            self.mask_candidate = np.vstack(candidate_list)
            print('using regular, mask_candidate shape = ', self.mask_candidate.shape)

    def __call__(self, image):

        mask = self.mask_candidate.copy()
        np.random.shuffle(mask)
        mask = rearrange(mask, '(d h w) (p1 p2 p3) -> (d p1) (h p2) (w p3)',
                         h=self.height // 2, w=self.width // 2, d=self.depth // 1, p1=1, p2=2, p3=2)
        mask = mask.flatten()
        _, d, w, h = image.shape
        image_seq = rearrange(image[0], '(h1 p1) (h2 p2) (h3 p3) ->  (h1 h2 h3) (p1 p2 p3) ',
                        p1=d//self.depth, p2=w//self.width, p3=h//self.height, h1=self.depth, h2=self.width, h3=self.height)

        index = np.argwhere(mask == 0).flatten()
        masked_image = image_seq #* mask
        masked_image = np.delete(masked_image, index, axis=0)
        image = rearrange(masked_image, '(h1 h2 h3) (p1 p2 p3) -> (h1 p1) (h2 p2) (h3 p3)',
                              p1=d // self.depth, p2=w // self.width, p3=h // self.height, h1=self.depth//1, h2=self.width//2,
                              h3=self.height//2)
        image = image[np.newaxis, :]
        return image
    
class Dataset3D(Dataset):
    def __init__(self, root, list_path, global_crop_size=(16, 128, 128), local_crop_size=(16, 64, 64), local_crops_number=0):

        self.root = root
        self.list_path = root + list_path
        self.img_ids = [i_id.strip().split() for i_id in open(self.list_path)]
        self.local_crops_number = local_crops_number

        self.files = []
        for item in self.img_ids:
            image_path = item
            name = image_path[0]
            img_file = image_path[0]
            self.files.append({
                "img": img_file,
                "DRR": img_file.replace("DL_patches_v2/", "DRR/").split("_dep")[0]+".npy",
                "name": name
            })
        print('SSL: {} images are loaded!'.format(len(self.img_ids)))

        self.global_crop3D_d, self.global_crop3D_h, self.global_crop3D_w = global_crop_size
        self.local_crop3D_d, self.local_crop3D_h, self.local_crop3D_w = local_crop_size
        self.tr_transforms3D_global0 = get_train_transform3D_global0()
        self.tr_transforms3D_global1 = get_train_transform3D_global1()
        self.tr_transforms3D_local = get_train_transform3D_local()
        self.mask_random = RandomMaskingGenerator(input_size=(self.global_crop3D_d//4,  self.global_crop3D_h//8, self.global_crop3D_w//8), mask_ratio=0.75)

    def __len__(self):
        return len(self.files)

    def truncate(self, CT):
        CT = CT / 1024.
        return CT

    def crop_scale0(self, image):
        _, _, img_h, img_w = image.shape

        scaler_h = np.random.uniform(1.4, 1.8)
        if (int(self.global_crop3D_h * scaler_h) >= img_h):
            scaler_h = 1.

        scaler_w = np.random.uniform(1.4, 1.8)
        if (int(self.global_crop3D_w * scaler_w) >= img_w):
            scaler_w = 1.

        scale_h = int(self.global_crop3D_h * scaler_h)
        scale_w = int(self.global_crop3D_w * scaler_w)

        h0 = random.randint(0, img_h - scale_h)
        w0 = random.randint(0, img_w - scale_w)

        h1 = h0 + scale_h
        w1 = w0 + scale_w

        image_crop = image[:, :, h0: h1, w0: w1]
        image_crop = self.truncate(image_crop)

        return image_crop

    def crop_scale_mirror_golbal(self, image, axes=(0, 1, 2)):
        _, img_d, img_h, img_w = image.shape

        scaler_d = 1.

        scaler_h = np.random.uniform(0.8, 1.2)
        if (int(self.global_crop3D_h * scaler_h) >= img_h):
            scaler_h = 1.

        scaler_w = np.random.uniform(0.8, 1.2)
        if (int(self.global_crop3D_w * scaler_w) >= img_w):
            scaler_w = 1.

        scale_d = int(self.global_crop3D_d * scaler_d)
        scale_h = int(self.global_crop3D_h * scaler_h)
        scale_w = int(self.global_crop3D_w * scaler_w)

        d0 = random.randint(0, img_d - scale_d)
        h0 = random.randint(0, img_h - scale_h)
        w0 = random.randint(0, img_w - scale_w)

        d1 = d0 + scale_d
        h1 = h0 + scale_h
        w1 = w0 + scale_w

        image_crop = image[:, d0: d1, h0: h1, w0: w1]

        if 2 in axes and np.random.uniform() < 0.8:
            image_crop = image_crop[:, :, :, ::-1]
        if 1 in axes and np.random.uniform() < 0.8:
            image_crop = image_crop[:, :, ::-1]
        if 0 in axes and np.random.uniform() < 0.8:
            image_crop = image_crop[:, ::-1]

        if (scaler_d != 1) or (scaler_w != 1) or (scaler_h != 1):
            image_crop = cv2.resize(image_crop[0].transpose(1, 2, 0), (self.global_crop3D_h, self.global_crop3D_w),
                                    interpolation=cv2.INTER_LINEAR)
            image_crop = image_crop[np.newaxis, :].transpose(0, 3, 1, 2)

        return image_crop

    def crop_scale_mirror_local(self, image, axes=(0, 1, 2)):
        _, img_d, img_h, img_w = image.shape

        scaler_d = 1.

        scaler_h = np.random.uniform(1.5, 2.0)
        if (int(self.local_crop3D_h * scaler_h) >= img_h):
            scaler_h = 1.

        scaler_w = np.random.uniform(1.5, 2.0)
        if (int(self.local_crop3D_w * scaler_w) >= img_w):
            scaler_w = 1.

        scale_d = int(self.local_crop3D_d * scaler_d)
        scale_h = int(self.local_crop3D_h * scaler_h)
        scale_w = int(self.local_crop3D_w * scaler_w)

        d0 = random.randint(0, img_d - scale_d)
        h0 = random.randint(0, img_h - scale_h)
        w0 = random.randint(0, img_w - scale_w)

        d1 = d0 + scale_d
        h1 = h0 + scale_h
        w1 = w0 + scale_w

        image_crop = image[:, d0: d1, h0: h1, w0: w1]

        if 2 in axes and np.random.uniform() < 0.8:
            image_crop = image_crop[:, :, :, ::-1]
        if 1 in axes and np.random.uniform() < 0.8:
            image_crop = image_crop[:, :, ::-1]
        if 0 in axes and np.random.uniform() < 0.8:
            image_crop = image_crop[:, ::-1]

        if (scaler_d != 1) or (scaler_w != 1) or (scaler_h != 1):
            image_crop = cv2.resize(image_crop[0].transpose(1, 2, 0), (self.local_crop3D_h, self.local_crop3D_w),
                                    interpolation=cv2.INTER_LINEAR)
            image_crop = image_crop[np.newaxis, :].transpose(0, 3, 1, 2)

        return image_crop

    def pad_image(self, img):

        """Pad an image up to the target size."""
        rows_missing = math.ceil(self.global_crop3D_w - img.shape[0])
        cols_missing = math.ceil(self.global_crop3D_h - img.shape[1])
        dept_missing = math.ceil(self.global_crop3D_d - img.shape[2])
        if rows_missing < 0:
            rows_missing = 0
        if cols_missing < 0:
            cols_missing = 0
        if dept_missing < 0:
            dept_missing = 0

        padded_img = np.pad(img, (
        (0, rows_missing), (0, cols_missing), (dept_missing // 2, dept_missing - dept_missing // 2)), 'constant')
        return padded_img

    def __getitem__(self, index):

        datafiles = self.files[index]
        # read nii file
        imageNII = nib.load(self.root + datafiles["img"])
        image = imageNII.get_fdata()
        name = datafiles["name"]
        image = self.pad_image(image)
        image = image[np.newaxis, :]
        image = image.transpose((0, 3, 1, 2))

        img = []

        DRR = cv2.resize(np.load(self.root + datafiles["DRR"]), (self.global_crop3D_h, self.global_crop3D_w),
                        interpolation=cv2.INTER_LINEAR)[np.newaxis, :]

        image_crop_ori = self.crop_scale0(image)

        # Global patches
        image_crop1 = self.crop_scale_mirror_golbal(image_crop_ori, axes=(0, 1, 2))
        image_crop2 = self.crop_scale_mirror_golbal(image_crop_ori, axes=(0, 1, 2))

        data_dict1 = {'image': image_crop1.astype(np.float32).copy(), 'label': None, 'name': name}
        data_dict2 = {'image': image_crop2.astype(np.float32).copy(), 'label': None, 'name': name}
        data_dict3 = {'image': DRR.astype(np.float32).copy(), 'label': None, 'name': name}
        data_dict1 = self.tr_transforms3D_global0(**data_dict1)
        data_dict2 = self.tr_transforms3D_global1(**data_dict2)

        img.append(self.mask_random(data_dict1['image']))
        img.append(self.mask_random(data_dict2['image']))
        img.append(data_dict3['image'])

        return img


def get_train_transform3D_global0():
    tr_transforms = []
    tr_transforms.append(GaussianNoiseTransform(data_key="image"))
    tr_transforms.append(
        GaussianBlurTransform(blur_sigma=(0.1, 2.), different_sigma_per_channel=True, p_per_channel=0.8,
                              p_per_sample=0.8, data_key="image"))
    tr_transforms.append(BrightnessMultiplicativeTransform((0.75, 1.25), p_per_sample=0.8, data_key="image"))
    tr_transforms.append(BrightnessTransform(0.0, 0.4, True, p_per_sample=0.8, p_per_channel=0.8, data_key="image"))
    tr_transforms.append(ContrastAugmentationTransform(data_key="image"))
    tr_transforms.append(GammaTransform(gamma_range=(0.7, 1.5), invert_image=False, per_channel=True, retain_stats=True,
                                        p_per_sample=0.8, data_key="image"))

    # now we compose these transforms together
    tr_transforms = Compose(tr_transforms)
    return tr_transforms


def get_train_transform3D_global1():
    tr_transforms = []
    tr_transforms.append(GaussianNoiseTransform(data_key="image"))
    tr_transforms.append(
        GaussianBlurTransform(blur_sigma=(0.1, 2.), different_sigma_per_channel=True, p_per_channel=0.5,
                              p_per_sample=0.8, data_key="image"))
    tr_transforms.append(BrightnessMultiplicativeTransform((0.75, 1.25), p_per_sample=1.0, data_key="image"))
    tr_transforms.append(BrightnessTransform(0.0, 0.4, True, p_per_sample=0.5, p_per_channel=0.8, data_key="image"))
    tr_transforms.append(ContrastAugmentationTransform(data_key="image"))
    tr_transforms.append(GammaTransform(gamma_range=(0.7, 1.5), invert_image=False, per_channel=True, retain_stats=True,
                                        p_per_sample=1.0, data_key="image"))

    # now we compose these transforms together
    tr_transforms = Compose(tr_transforms)
    return tr_transforms


def get_train_transform3D_local():
    tr_transforms = []
    tr_transforms.append(GaussianNoiseTransform(data_key="image"))
    tr_transforms.append(
        GaussianBlurTransform(blur_sigma=(0.1, 2.), different_sigma_per_channel=True, p_per_channel=0.8,
                              p_per_sample=0.5, data_key="image"))
    tr_transforms.append(BrightnessMultiplicativeTransform((0.75, 1.25), p_per_sample=1.0, data_key="image"))
    tr_transforms.append(BrightnessTransform(0.0, 0.4, True, p_per_sample=0.8, p_per_channel=0.8, data_key="image"))
    tr_transforms.append(ContrastAugmentationTransform(data_key="image"))
    tr_transforms.append(GammaTransform(gamma_range=(0.7, 1.5), invert_image=False, per_channel=True, retain_stats=True,
                                        p_per_sample=0.8, data_key="image"))

    # now we compose these transforms together
    tr_transforms = Compose(tr_transforms)
    return tr_transforms

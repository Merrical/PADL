# -*- coding: utf-8 -*-
import os
import torch
import PIL.Image

import numpy as np
import pandas as pd
from torch.utils import data
from batchgenerators.dataloading.data_loader import DataLoader
from batchgenerators.transforms.abstract_transforms import Compose
from batchgenerators.transforms.spatial_transforms import SpatialTransform_2, MirrorTransform
from batchgenerators.transforms.color_transforms import BrightnessMultiplicativeTransform, GammaTransform
from batchgenerators.transforms.noise_transforms import GaussianNoiseTransform, GaussianBlurTransform

from dataset.config import Config


config = Config()


class Disc_Cup(data.Dataset):  # inherit
    """
    load data in a folder
    """

    def __init__(self, root, batch_size, DF, transform=True):
        super(Disc_Cup, self).__init__()
        self.root = root
        self._transform = transform
        self.scale_size = config.SCALE_SIZE
        self.batch_size = batch_size

        self.DF = pd.DataFrame(columns=['imgName', 'maskName', 'label', 'center', 'xmin', 'ymin',
                                        'xmax', 'ymax', 'width', 'height', 'discFlag', 'rater'])
        for spilt in DF:
            DF_all = pd.read_csv(root + '/' + 'Glaucoma_multirater_' + spilt + '.csv', encoding='gbk')

            DF_this = DF_all.loc[DF_all['rater'] == 0]      # Final Label
            DF_this = DF_this.reset_index(drop=True)
            DF_this = DF_this.drop('Unnamed: 0', 1)
            self.DF = pd.concat([self.DF, DF_this])

        self.DF.index = range(0, len(self.DF))

    def __len__(self):
        return len(self.DF)

    def __getitem__(self, index):
        img_Name = self.DF.loc[index, 'imgName']

        """ Get the images """
        fullPathName = os.path.join(self.root, img_Name)
        fullPathName = fullPathName.replace('\\', '/')  # image path

        img = PIL.Image.open(fullPathName).convert('RGB')  # read image
        img = img.resize((self.scale_size, self.scale_size))
        img = np.array(img)

        if len(img.shape) == 2:
            img = np.expand_dims(img, axis=2)    # add additional channel in dim 2 (channel)

        img_ori = img

        """ Get the six raters masks """
        masks = []
        data_path = self.root
        for n in range(1, 7):     # n:1-6
            # # load rater 1-6 label recurrently
            maskName = self.DF.loc[index, 'maskName'].replace('FinalLabel','Rater'+str(n))
            fullPathName = os.path.join(data_path, maskName)
            fullPathName = fullPathName.replace('\\', '/')

            Mask = PIL.Image.open(fullPathName).convert('L')
            Mask = Mask.resize((self.scale_size, self.scale_size))
            Mask = np.array(Mask)

            if Mask.max() > 1:
                Mask = Mask / 255.0

            disc = Mask.copy()
            disc[disc != 0] = 1
            cup = Mask.copy()
            cup[cup != 1] = 0
            Mask = np.stack((disc,cup))

            # Mask = Mask.transpose((2, 0, 1))
            Mask = torch.from_numpy(Mask)
            masks.append(Mask)

        if self._transform:
            img_ori, img, masks = self.transform(img_ori, img, masks)
            return {'image': img, 'image_ori': img_ori, 'mask': masks, 'name': img_Name.split('.')[0]}
        else:
            return {'image': img, 'image_ori': img_ori, 'mask': masks, 'name': img_Name.split('.')[0]}

    # Translating numpy_array into format that pytorch can use on Code.
    def transform(self, img_o, img, lbl):
        if img.max() > 1:
            img = img.astype(np.float64) / 255.0
        img -= config.MEAN_AND_STD['mean_rgb']
        img /= config.MEAN_AND_STD['std_rgb']
        img = img.transpose(2, 0, 1)
        img = torch.from_numpy(img)

        if img.max() > 1:
            img_o = img_o.astype(np.float64) / 255.0
        img_o = img_o.transpose(2, 0, 1)
        img_o = torch.from_numpy(img_o)

        return img_o, img, lbl


class Disc_Cup_DataLoader(DataLoader):
    def __init__(self, data, batch_size, patch_size, num_threads_in_multithreaded, seed_for_shuffle=1234,
                 return_incomplete=False, shuffle=True):
        super().__init__(data, batch_size, num_threads_in_multithreaded, seed_for_shuffle, return_incomplete,
                         shuffle, True)
        self.patch_size = patch_size
        self.num_modalities = 3  # == channels
        self.indices = list(range(len(data.DF)))  # data --> dataset

    def generate_train_batch(self):
        idx = self.get_indices()  # len = batch_size
        samples_for_batch = [self._data[i] for i in idx]
        # self._data[i]['image']: torch.size([3,256,256]);
        # self._data[i]['mask']: list(6), item.size = torch.size([2,256,256])
        data = np.zeros((self.batch_size, self.num_modalities, *self.patch_size), dtype=np.float32)
        mask1 = np.zeros((self.batch_size, 2, *self.patch_size), dtype=np.float32)
        mask2 = np.zeros((self.batch_size, 2, *self.patch_size), dtype=np.float32)
        mask3 = np.zeros((self.batch_size, 2, *self.patch_size), dtype=np.float32)
        mask4 = np.zeros((self.batch_size, 2, *self.patch_size), dtype=np.float32)
        mask5 = np.zeros((self.batch_size, 2, *self.patch_size), dtype=np.float32)
        mask6 = np.zeros((self.batch_size, 2, *self.patch_size), dtype=np.float32)

        for i, data_temp in enumerate(samples_for_batch):
            data[i] = data_temp['image'].to(dtype=torch.float32).numpy()
            mask1[i] = data_temp['mask'][0].numpy()
            mask2[i] = data_temp['mask'][1].numpy()
            mask3[i] = data_temp['mask'][2].numpy()
            mask4[i] = data_temp['mask'][3].numpy()
            mask5[i] = data_temp['mask'][4].numpy()
            mask6[i] = data_temp['mask'][5].numpy()
        data, mask1, mask2, mask3, mask4, mask5, mask6 = torch.from_numpy(data), torch.from_numpy(mask1), \
                                                         torch.from_numpy(mask2), torch.from_numpy(mask3), \
                                                         torch.from_numpy(mask4), torch.from_numpy(mask5), \
                                                         torch.from_numpy(mask6)
        mask = [mask1, mask2, mask3, mask4, mask5, mask6]
        return {'image': data, 'mask': mask}


def get_train_transform(patch_size):
    # we now create a list of transforms. These are not necessarily the best transforms to use for BraTS, this is just
    # to showcase some things
    tr_transforms = []

    # the first thing we want to run is the SpatialTransform. It reduces the size of our data to patch_size and thus
    # also reduces the computational cost of all subsequent operations. All subsequent operations do not modify the
    # shape and do not transform spatially, so no border artifacts will be introduced
    # Here we use the new SpatialTransform_2 which uses a new way of parameterizing elastic_deform
    # We use all spatial transformations with a probability of 0.2 per sample. This means that 1 - (1 - 0.1) ** 3 = 27%
    # of samples will be augmented, the rest will just be cropped
    tr_transforms.append(
        SpatialTransform_2(
            patch_size, [i // 2 for i in patch_size],
            do_elastic_deform=True, deformation_scale=(0, 0.25),
            do_rotation=True,
            angle_x=(- 15 / 360. * 2 * np.pi, 15 / 360. * 2 * np.pi),
            angle_y=(- 15 / 360. * 2 * np.pi, 15 / 360. * 2 * np.pi),
            angle_z=(- 15 / 360. * 2 * np.pi, 15 / 360. * 2 * np.pi),
            do_scale=True, scale=(0.75, 1.25),
            border_mode_data='constant', border_cval_data=0,
            border_mode_seg='constant', border_cval_seg=0,
            order_seg=1, order_data=3,
            random_crop=True,
            p_el_per_sample=0.1, p_rot_per_sample=0.1, p_scale_per_sample=0.1
        )
    )

    # now we mirror along all axes
    tr_transforms.append(MirrorTransform(axes=(0, 1, 2)))

    # brightness transform for 15% of samples
    tr_transforms.append(BrightnessMultiplicativeTransform((0.7, 1.5), per_channel=True, p_per_sample=0.15))

    # gamma transform. This is a nonlinear transformation of intensity values
    # (https://en.wikipedia.org/wiki/Gamma_correction)
    tr_transforms.append(GammaTransform(gamma_range=(0.5, 2), invert_image=False, per_channel=True, p_per_sample=0.15))
    # we can also invert the image, apply the transform and then invert back
    tr_transforms.append(GammaTransform(gamma_range=(0.5, 2), invert_image=True, per_channel=True, p_per_sample=0.15))

    # Gaussian Noise
    tr_transforms.append(GaussianNoiseTransform(noise_variance=(0, 0.05), p_per_sample=0.15))

    # blurring. Some BraTS cases have very blurry modalities. This can simulate more patients with this problem and
    # thus make the model more robust to it
    tr_transforms.append(GaussianBlurTransform(blur_sigma=(0.5, 1.5), different_sigma_per_channel=True,
                                               p_per_channel=0.5, p_per_sample=0.15))

    # now we compose these transforms together
    tr_transforms = Compose(tr_transforms)
    return tr_transforms

#
# if __name__ == "__main__":
#     train_set = Disc_Cup('/media/userdisk0/Dataset/DiscRegion', DF=['BinRushed', 'MESSIDOR'], transform=True)
#     batch_size = 32
#     patch_size = (256, 256)
#     train_loader = Disc_Cup_DataLoader(train_set, batch_size, patch_size, 1)
#     batch = next(train_loader)
#
#     tr_transforms = get_train_transform(patch_size)
#     train_gen = MultiThreadedAugmenter(train_loader, tr_transforms, num_processes=4, num_cached_per_queue=3,
#     seeds=None, pin_memory=False)
#     train_gen.restart()
#     batch = next(train_gen)

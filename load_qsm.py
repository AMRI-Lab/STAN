from genericpath import exists
import os
from imageio import save
import torch
import numpy as np
import pandas as pd
from scipy.ndimage import binary_fill_holes 
from torch.utils.data import Dataset
import nibabel as nib
import random
import torch.nn.functional as Func
from monai import transforms

def nii_loader(path):
    img = nib.load(str(path))
    data = img.get_data()
    return data

def read_table(path):
    return(pd.read_excel(path).values)


def crop(image, mask):

    nonzero_mask = image != 0
    nonzero_mask = binary_fill_holes(nonzero_mask)
    mask_voxel_coords = np.where(nonzero_mask != 0)
    minzidx = int(np.min(mask_voxel_coords[0]))
    maxzidx = int(np.max(mask_voxel_coords[0])) + 1
    minxidx = int(np.min(mask_voxel_coords[1]))
    maxxidx = int(np.max(mask_voxel_coords[1])) + 1
    minyidx = int(np.min(mask_voxel_coords[2]))
    maxyidx = int(np.max(mask_voxel_coords[2])) + 1
    bbox = [[minzidx, maxzidx], [minxidx, maxxidx], [minyidx, maxyidx]]

    resizer = (slice(bbox[0][0], bbox[0][1]), slice(bbox[1][0], bbox[1][1]), slice(bbox[2][0], bbox[2][1]))
    cropped_image = image[resizer]
    if mask is not None:
        cropped_mask = mask[resizer]
    else:
        cropped_mask = None
    return cropped_image, cropped_mask, np.array(bbox)

def dice_coeff(y_pred, y_true, num_classes):
    y_pred = Func.softmax(y_pred, dim = 1)
    eps = 1.
    dice = 0
    for c in range(num_classes):
        if c == 0:
            continue
        jaccard_target = (y_true == c).float()
        jaccard_output = y_pred[:, c]
        intersection = (jaccard_output * jaccard_target).sum()
        union = jaccard_output.sum() + jaccard_target.sum()
        dice += (((2 * intersection + eps) / (union + eps)) / (num_classes - 1))
    return dice

size = (80, 96, 80)



train_aug = transforms.Compose(
        [
            transforms.AddChanneld(keys=["image", "label"]),
            # transforms.Resized(keys=["image", "label"], spatial_size=size, mode=['trilinear', 'nearest'], align_corners=[True, None]),
            transforms.RandFlipd(keys=["image", "label"], prob=0.2, spatial_axis=0),
            transforms.RandFlipd(keys=["image", "label"], prob=0.2, spatial_axis=1),
            transforms.RandFlipd(keys=["image", "label"], prob=0.2, spatial_axis=2),
            transforms.RandRotate90d(keys=["image", "label"], prob=0.2, max_k=3, spatial_axes=(0,1)),
            transforms.RandRotated(keys=["image", "label"], range_x=0.5, range_y=0.5, range_z=0.5, prob=0.1, keep_size=True, 
                mode='bilinear', padding_mode='border', align_corners=False),
            # transforms.RandCropByLabelClasses(keys=["image", "label"], label_key="label", spatial_size=[3, 3], 
            #     ratios=[1, 2, 2, 2, 2, 2, 2, 2, 2], num_classes=9, num_samples=1)
            # transforms.RandSpatialCropd(keys=["image", "label"], roi_size=min_size, max_roi_size=None, random_center=True, random_size=True, allow_missing_keys=False),
            transforms.RandAffined(keys=["image", "label"], spatial_size=None, prob=0.1, rotate_range=None, shear_range=(0.5, 0.5), 
                translate_range=None, scale_range=None, mode='bilinear', padding_mode='reflection', allow_missing_keys=False),
            # transforms.RandScaleIntensityd(keys="image", factors=0.1, prob=0.2),
            # transforms.RandGaussianSmoothd(keys="image", sigma_x=(0.25, 1.5), sigma_y=(0.25, 1.5), sigma_z=(0.25, 1.5), prob=0.1, approx='erf'),
            # transforms.RandZoomd(keys=["image", "label"], prob=0.1, min_zoom=0.8, max_zoom=1.2, mode=['trilinear', 'nearest'], align_corners=[True, None], padding_mode="edge"),
            # transforms.RandAdjustContrastd(keys="image", prob=0.1, gamma=(0.5, 4.5), allow_missing_keys=False),
            # transforms.RandShiftIntensityd(keys="image", offsets=0.1, prob=0.2),
            # transforms.Rand3DElasticd(keys=["image", "label"], sigma_range=(5, 7), magnitude_range=(50, 150), prob=0.1),            
        ]
    )

test_aug = transforms.Compose(
        [
            transforms.AddChanneld(keys=["image", "label"])     
        ]
    )


class TrainDataset(Dataset):
    def __init__(self, excel_path, img_root, mask_root, file_list, num_class, loader=nii_loader, table_reader=read_table, transform=train_aug):
        self.img_root = img_root
        self.mask_root = mask_root
        self.file_list = file_list
        self.loader = loader
        self.table = table_reader(excel_path)
        self.transform = transform
        self.num_class = num_class
    
    def __getitem__(self, index):
        file_name = self.file_list[index]
        age = None
        for f in self.table:
            sid = str(f[0])
            if sid not in file_name:
                continue
            age = int(f[2])
            gender = int(f[1])
        if age is None:
            print("age is none ", file_name)
            

        data_list = os.listdir(os.path.join(self.img_root, file_name))
        for f in data_list:
            if f.endswith(".nii") or f.endswith(".nii.gz"):
                img_path = os.path.join(self.img_root, file_name, f)
        img = self.loader(img_path)
        # img_path = os.path.join(self.img_root, file_name)
        # img = self.loader(img_path)
        
        # mask_list = os.listdir(os.path.join(self.mask_root, file_name))
        # for f in mask_list:
        #     mask_path = os.path.join(self.mask_root, file_name + '.nii.gz')
        mask_path = os.path.join(self.mask_root, file_name)
        mask = self.loader(mask_path)
        # mask = None
        
        img, mask, bbox = crop(img, mask)
        if mask is None:
            img = self.transform(img)
        else:
            data = {'image': img, 'label': mask}
            aug_data = self.transform(data)
            img, mask = aug_data['image'], aug_data['label']
            
        img = np.expand_dims(img, axis=(0))
        img = np.ascontiguousarray(img, dtype= np.float32) 
        img = torch.from_numpy(img).type(torch.FloatTensor)
        img = Func.interpolate(img, size, mode = 'trilinear', align_corners = True)
        img = img[0]

        if mask is not None:
            mask = np.expand_dims(mask, axis=(0))
            mask = np.ascontiguousarray(mask, dtype= np.float32)
            mask = torch.from_numpy(mask).type(torch.FloatTensor)
            mask = Func.interpolate(mask, size, mode = 'nearest')
            mask = mask[0, 0]

        if mask is None:
            return img, file_name, age, gender, bbox
        else:
            return img, mask, file_name, age, gender, bbox
    
    def __len__(self):
        return len(self.file_list)
    

def SaveResult(mask_root, save_path, img, name, bbox):
    
    print(name, img.shape)
    img_out = np.zeros(img[0].shape)
    img_out = np.argmax(img, axis = 0)
    img = img_out.astype(np.uint8)

    # mask_list = os.listdir(os.path.join(mask_root, name))
    # for f in mask_list:
    # mask_path = os.path.join(mask_root, name)# + .nii.gz")
    mask_path = os.path.join(mask_root, name)
    mask = nib.load(mask_path)
    affine = mask.affine
    header = mask.header

    real_size = (bbox[0][1] - bbox[0][0], bbox[1][1] - bbox[1][0], bbox[2][1] - bbox[2][0])
    out = np.zeros(mask.shape)
    img = np.expand_dims(img, axis=(0,1))
    img = np.ascontiguousarray(img, dtype= np.float32)
    img = torch.from_numpy(img).type(torch.FloatTensor)
    img = Func.interpolate(img, real_size, mode = 'nearest')
    img = img[0, 0].data.numpy().astype(np.uint8)
    resizer = (slice(bbox[0][0], bbox[0][1]), slice(bbox[1][0], bbox[1][1]), slice(bbox[2][0], bbox[2][1]))
    out[resizer] = img
    np.round(out)

    new_img = nib.Nifti1Image(out.astype(np.uint8), affine = affine, header = header)
    nib.save(new_img, os.path.join(save_path, name+'.nii.gz'))
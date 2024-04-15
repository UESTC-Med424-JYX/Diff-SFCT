import os
import random

import nibabel as nib
import numpy as np
import torch
from scipy import ndimage
from scipy.ndimage.interpolation import zoom
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


def to_one_hot(labels, num_classes):
    labels = labels.clone().detach().long()
    one_hot = torch.zeros(num_classes, labels.size(0), labels.size(1))
    one_hot = one_hot.scatter(0, labels.unsqueeze(0), 1)[1:]
    return one_hot


def random_rot_flip(image, label):
    k = np.random.randint(0, 4)
    image = np.rot90(image, k)
    label = np.rot90(label, k)
    axis = np.random.randint(0, 2)
    image = np.flip(image, axis=axis).copy()
    label = np.flip(label, axis=axis).copy()
    return image, label


def random_rotate(image, label):
    angle = np.random.randint(-20, 20)
    image = ndimage.rotate(image, angle, order=0, reshape=False)
    label = ndimage.rotate(label, angle, order=0, reshape=False)
    return image, label


class RandomGenerator(object):
    def __init__(self, output_size, num_classes):
        self.output_size = output_size
        self.num_classes = num_classes

    def __call__(self, sample):

        image, label = sample['image'], sample['label']

        if random.random() > 0.5:
            image, label = random_rot_flip(image, label)
        elif random.random() > 0.5:
            image, label = random_rotate(image, label)

        x, y = image.shape
        if x != self.output_size[0] or y != self.output_size[1]:
            image = zoom(image, (self.output_size[0] / x, self.output_size[1] / y), order=3)  # why not 3?
            label = zoom(label, (self.output_size[0] / x, self.output_size[1] / y), order=0)

        image = torch.from_numpy(image.astype(np.float32)).unsqueeze(0)
        label = torch.from_numpy(label.astype(np.float32))

        label = to_one_hot(label, self.num_classes)
        sample = {'image': image, 'label': label.long()}
        return sample


class ValidGenerator(object):
    def __init__(self, output_size, num_classes):
        self.output_size = output_size
        self.num_classes = num_classes

    def __call__(self, sample):
        image, label = sample['image'], sample['label']

        x, y = image.shape
        image = zoom(
            image, (self.output_size[0] / x, self.output_size[1] / y), order=3)
        label = zoom(
            label, (self.output_size[0] / x, self.output_size[1] / y), order=0)

        image = torch.from_numpy(
            image.astype(np.float32)).unsqueeze(0)

        label = torch.from_numpy(label.astype(np.uint8))

        label = to_one_hot(label, self.num_classes)

        sample = {'image': image, 'label': label}
        return sample


class ACDCDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.annotation_lines = []

        for dir in os.listdir(data_dir):
            for file in os.listdir(os.path.join(data_dir, dir)):

                if file.endswith("_gt_.nii.gz"):
                    filename = file.split("_gt")[0]
                    self.annotation_lines.append(filename)

        self.data_dir = data_dir
        self.transform = transform

    def __len__(self):
        return len(self.annotation_lines)

    def __getitem__(self, index):
        name = self.annotation_lines[index] + '.nii.gz'

        patient_dir = name.split("_frame")[0]

        img_path = os.path.join(self.data_dir, patient_dir, name)
        mask_name = self.annotation_lines[index] + '_gt_.nii.gz'
        mask_path = os.path.join(self.data_dir, patient_dir, mask_name)

        image = nib.load(img_path).get_fdata()
        label = nib.load(mask_path).get_fdata()

        sample = {
            "image": image,
            "label": label
        }

        if self.transform:
            state = torch.get_rng_state()
            torch.set_rng_state(state)
            sample = self.transform(sample)

        sample["file"] = self.annotation_lines[index]

        return sample


def get_loader_acdc(train_dir, test_dir, image_size, num_classes):
    transform_train = transforms.Compose([
        RandomGenerator([image_size, image_size], num_classes),
    ])
    transform_valid = transforms.Compose([
        ValidGenerator([image_size, image_size], num_classes),
    ])
    train_ds = ACDCDataset(os.path.join(train_dir), transform=transform_train)
    valid_ds = ACDCDataset(os.path.join(test_dir), transform=transform_valid)

    return [train_ds, valid_ds]


if __name__ == '__main__':

    train_dir = "/data/jyx/ACDC/database/jyx_patient/base"
    test_dir = "/data/jyx/ACDC/database/jyx_patient/testing"

    image_size = 224
    batch_size = 4

    [train_ds, test_ds] = get_loader_acdc(train_dir, test_dir, image_size, 4)

    print("train_ds.length   ===>   ", len(train_ds))
    print("test_ds.length   ===>   ", len(test_ds))

    train_loader = DataLoader(train_ds, batch_size=batch_size)

    for idx, batch in enumerate(train_loader):
        image, label = batch['image'], batch['label']
        file = batch['file']

        print(file)
        print(image.shape)
        print(label.shape)

        print()

        break

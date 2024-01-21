import torch
import numpy as np
import glob as glob

from torch.utils.data import DataLoader, Dataset
from PIL import Image

TRAIN_BATCH_SIZE = 128
TEST_BATCH_SIZE = 1

# The SRCNN dataset module.


class SRCNNDataset(Dataset):
    def __init__(self, image_paths, label_paths):
        self.all_image_paths = glob.glob(f"{image_paths}/*")
        self.all_label_paths = glob.glob(f"{label_paths}/*")

    def __len__(self):
        return (len(self.all_image_paths))

    def __getitem__(self, index):
        # 原本圖片是 RGB ，但是在大圖片轉成高解析度與低解析度的小 patch 時換成 BRG 了，
        # 現在(train前)又轉回雲本的 RGB。
        image = Image.open(self.all_image_paths[index]).convert('RGB')
        label = Image.open(self.all_label_paths[index]).convert('RGB')

        image = np.array(image, dtype=np.float32)
        label = np.array(label, dtype=np.float32)

        # 原本的大圖與小 patch 都沒有正規化，這裡把資料與標籤都除以 255.
        image /= 255.
        label /= 255.

        # p.s., 原本圖片跟 patch, 其 np.array(object).shape = (height, width, 3)
        # bring the channel dimension to the front
        image = image.transpose([2, 0, 1])  # (3, height, width)
        label = label.transpose([2, 0, 1])

        return (
            torch.tensor(image, dtype=torch.float),
            torch.tensor(label, dtype=torch.float)
        )

# < Note : Dataset c.f. Dataloader >---------------
# https://stackoverflow.com/questions/61562456/problem-with-dataloader-object-not-subscriptable
#
# 1. torch.utils.data.Dataset object is indexable.
# 2. torch.utils.data.DataLoader - non-indexable, only iterable,
#    usually returns batches of data from above Dataset.
# -------------------------------------------------


# Prepare the datasets.

def get_datasets(
    train_image_paths, train_label_paths,
    valid_image_path, valid_label_paths
):
    dataset_train = SRCNNDataset(
        train_image_paths, train_label_paths
    )
    dataset_valid = SRCNNDataset(
        valid_image_path, valid_label_paths
    )
    return dataset_train, dataset_valid

# Prepare the data loaders


def get_dataloaders(dataset_train, dataset_valid):
    train_loader = DataLoader(
        dataset_train,
        batch_size=TRAIN_BATCH_SIZE,
        shuffle=True
    )
    valid_loader = DataLoader(
        dataset_valid,
        batch_size=TEST_BATCH_SIZE,
        shuffle=False
    )
    dataiter = iter(train_loader)
    images, labels = next(dataiter)

    # < How to find shape and columns for dataloader? > --------------------
    # https://discuss.pytorch.org/t/how-to-find-shape-and-columns-for-dataloader/34901
    # Method 1 :
    print(type(images))  # <class 'torch.Tensor'>
    print(images.shape)  # torch.Size([128, 3, 32, 32])
    print(labels.shape)  # torch.Size([128, 3, 32, 32])

    # Method 2 :
    for i, (images, labels) in enumerate(train_loader):
        print(type(images))
        print(images.shape)  # torch.Size([128, 3, 32, 32])
        print(labels.shape)  # torch.Size([128, 3, 32, 32])
        break
    # -----------------------------------------------------

    return train_loader, valid_loader


if __name__ == '__main__':
    # Constants
    TRAIN_LABEL_PATHS = '../input/t91_hr_patches'
    TRAN_IMAGE_PATHS = '../input/t91_lr_patches'
    VALID_LABEL_PATHS = '../input/test_hr'
    VALID_IMAGE_PATHS = '../input/test_bicubic_rgb_2x'

    dataset_train, dataset_valid = get_datasets(
        TRAN_IMAGE_PATHS, TRAIN_LABEL_PATHS,
        VALID_IMAGE_PATHS, VALID_LABEL_PATHS
    )
    get_dataloaders(dataset_train, dataset_valid)

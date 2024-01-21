from PIL import Image  # for reading and resizing images
from tqdm import tqdm

import matplotlib.pyplot as plt
import patchify
import numpy as np
import matplotlib.gridspec as gridspec
import glob as glob
import os
import cv2  # for saving the patches to disk


# "SHOW_PATCHES" is better to keep it as False after visualizing a few patches
# as you will have to press a key on the keyboard every time an image pops up
SHOW_PATCHES = False
STRIDE = 14  # 14 has the arguments.
SIZE = 32


def show_patches(patches):
    """
     stores a boolean value indicating
     whether we want to visualize the image patches while executing the code or not
    """
    plt.figure(figsize=(patches.shape[0], patches.shape[1]))
    gs = gridspec.GridSpec(patches.shape[0], patches.shape[1])
    gs.update(wspace=0.01, hspace=0.02)
    counter = 0
    for i in range(patches.shape[0]):
        for j in range(patches.shape[1]):
            ax = plt.subplot(gs[counter])
            plt.imshow(patches[i, j, 0, :, :, :])
            plt.axis('off')
            counter += 1
    plt.show()


def create_patches(
    input_paths, out_hr_path, out_lr_path,
):
    os.makedirs(out_hr_path, exist_ok=True)
    os.makedirs(out_lr_path, exist_ok=True)
    all_paths = []

    for input_path in input_paths:
        # read and store all the original image patches
        # extend: 取出 object 的所有 element/iterator 扔進list 裡面
        all_paths.extend(glob.glob(f"{input_path}/*"))
    print(f"Creating patches for {len(all_paths)} images")

    for image_path in tqdm(all_paths, total=len(all_paths)):
        # read the images
        image = Image.open(image_path)
        # t1, t2, t3, ...
        image_name = image_path.split(os.path.sep)[-1].split('.')[0]

        # extract the original height and width
        # e.g., image.size for ../T91/t1.png = (197, 176)
        # But, np.array(image).shape = (176, 197, 3).
        # Notice that the indices are different.
        w, h = image.size  # (197, 176)

        # Create patches of size (32, 32, 3)
        # It takes the input image as an array, the patch size that we want along with the color channel,
        # and the stride, which is 14 here as the arguments.
        patches = patchify.patchify(
            np.array(image), (32, 32, 3), STRIDE)  # patches : (11, 12, 1, 32, 32, 3)

        if SHOW_PATCHES:
            show_patches(patches)

        counter = 0
        for i in range(patches.shape[0]):
            for j in range(patches.shape[1]):
                counter += 1
                patch = patches[i, j, 0, :, :, :]
                # converting to BGR format
                patch = cv2.cvtColor(patch, cv2.COLOR_RGB2BGR)
                cv2.imwrite(
                    f"{out_hr_path}/{image_name}_{counter}.png",
                    patch
                )

                # Convert to bicubic and save.
                h, w, _ = patch.shape
                # low resolution patch
                low_res_img = cv2.resize(patch, (int(w*0.5), int(h*0.5)),
                                         interpolation=cv2.INTER_CUBIC)

                # Now upscale using BICUBIC.
                high_res_upscale = cv2.resize(low_res_img, (w, h),
                                              interpolation=cv2.INTER_CUBIC)
                cv2.imwrite(
                    f"{out_lr_path}/{image_name}_{counter}.png",
                    high_res_upscale
                )


if __name__ == '__main__':
    create_patches(
        ['../input/T91'],
        '../input/t91_hr_patches',
        '../input/t91_lr_patches'
    )

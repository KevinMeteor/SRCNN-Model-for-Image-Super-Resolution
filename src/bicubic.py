"""
This python script is for converting high-resolution images to low-resolution
bicubic images.
USAGE EXAMPLE:
python bicubic.py --path ../input/Set14/original ../input/Set5/original --scale-factor 2x
"""

from PIL import Image

import glob as glob
import os
import argparse

# Construct the argument parser and parse the command line arguments.
parser = argparse.ArgumentParser()
parser.add_argument('-p', '--path', default='../input/Set14/original',
                    # nargs : Number of times the argument can be used.
                    nargs='+',
                    help='path to the high-res images to convert to low-res')
parser.add_argument('-s', '--scale-factor', dest='scale_factor', default='2x',
                    help='make low-res by how much factor',
                    # choices : Limit values to a specific set of choices.
                    choices=['2x', '3x', '4x'])
args = vars(parser.parse_args())

paths = args['path']
images = []
for path in paths:
    images.extend(glob.glob(f"{path}/*.png"))
print(len(images))

# Select scaling-factor and set up directories according to that.
if args['scale_factor'] == '2x':
    scale_factor = 0.5
    os.makedirs('../input/test_bicubic_rgb_2x', exist_ok=True)
    save_path_lr = '../input/test_bicubic_rgb_2x'
    os.makedirs('../input/test_hr', exist_ok=True)
    save_path_hr = '../input/test_hr'
if args['scale_factor'] == '3x':
    scale_factor = 0.333
    os.makedirs('../input/test_bicubic_rgb_3x', exist_ok=True)
    os.makedirs('../input/test_hr', exist_ok=True)
    save_path_lr = '../input/test_bicubic_rgb_3x'
    save_path_hr = '../input/test_hr'
if args['scale_factor'] == '4x':
    scale_factor = 0.25
    # os.makedirs : Recursive directory creation function.
    # If exist_ok is False (the default),
    # a FileExistsError is raised if the target directory already exists.
    os.makedirs('../input/test_bicubic_rgb_4x', exist_ok=True)
    os.makedirs('../input/test_hr', exist_ok=True)
    save_path_lr = '../input/test_bicubic_rgb_4x'
    save_path_hr = '../input/test_hr'

print(f"Scaling factor: {args['scale_factor']}")
print(f"Low resolution images save path: {save_path_lr}")

for image in images:
    orig_img = Image.open(image)
    image_name = image.split(os.path.sep)[-1]
    w, h = orig_img.size[:]
    print(f"Original image dimensions: {w}, {h}")
    # original images save to '../input/test_hr'
    orig_img.save(f"{save_path_hr}/{image_name}")

    low_res_img = orig_img.resize(
        (int(w*scale_factor), int(h*scale_factor)), Image.BICUBIC)
    # Upscale using BICUBIC.
    high_res_upscale = low_res_img.resize((w, h), Image.BICUBIC)
    # blurred images save to '../input/test_bicubic_rgb_?x', where ? = 2, 3, 4.
    high_res_upscale.save(f"{save_path_lr}/{image_name}")

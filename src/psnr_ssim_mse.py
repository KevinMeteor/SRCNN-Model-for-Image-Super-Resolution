"""
20240120
計算各個模型圖片比較結果 : PSNR, SSIM, MSE

"""

# %%
from image_psnr_ssim_mse_func import compare_images_scores
import pandas as pd
from PIL import Image
import numpy as np
import glob as glob
import os
import cv2

# choose which model to be compute validation or test PSNR, SSIM, MSE.
model = 'SRCNN1'

os.makedirs('../input/test_hr', exist_ok=True)
os.makedirs('../outputs/valid_results/' + model +
            '_valid_results', exist_ok=True)

# validation image
paths_valid = ['../outputs/valid_results/' + model + '_valid_results']

images_valid = []
for path in paths_valid:
    images_valid .extend(glob.glob(f"{path}/*.png"))

# https://note.nkmk.me/en/python-list-str-select-replace/
# Extract strings that contain or do not contain a specific substring.
# 去除包含所有 'loss', 'psnr' 字眼的 elements
images_valid = [s for s in images_valid if (
    'psnr' not in s) and ('loss' not in s)]
# print(len(images_valid))


# images_valid 依照檔名末位數字排列 20240120
idxs = [s.split('_')[-1][:-4] for s in images_valid]

df_images_valid = pd.DataFrame({
    'images': images_valid,
    'idx': map(int, idxs)
}).sort_values(by='idx')

images_valid = df_images_valid['images'].to_list()

# %%

for image in images_valid:
    orig_img = Image.open(image)
    image_name = image.split(os.path.sep)[-1]


# %%
# Ground-true image
paths_ground_true = ['..\\input\\test_hr']
images_ground_true = []

# images 按照檔名之英文字母順序引入
for path in paths_ground_true:
    images_ground_true.extend(glob.glob(f"{path}/*.png"))

# https://note.nkmk.me/en/python-list-str-select-replace/
# Extract strings that contain or do not contain a specific substring.
images_ground_true = [s for s in images_ground_true if (
    'psnr' not in s) and ('loss' not in s)]


# %%
# compute PSNR, SSIM, MSE
assert len(images_valid) == len(
    images_ground_true), 'two image sets\' sizes are not match! '

images = np.array([
    images_ground_true,
    images_valid
]).T


Scores_p_s_m = np.zeros(shape=(len(images_valid), 3))

for i in range(len(images)):
    img_ground_true = np.array(Image.open(images_ground_true[i]))
    img_valid = np.array(Image.open(images_valid[i]))
    # PSNR, SSIM, MSE
    p, s, m = compare_images_scores(
        img1=img_ground_true,
        img2=img_valid,
        channel_axis=-1)
    Scores_p_s_m[i] = (p, s, m)

# os.path.sep : '\\' in window, but '/' in linux
# 避免因在不一樣系統開發而出錯
image_name = [s.split(os.path.sep)[-1][:-4] for s in images_ground_true]


# %%
# Save the results.
df_Scores_p_s_m = pd.DataFrame(
    Scores_p_s_m,
    index=image_name,
)

# Add column names
df_Scores_p_s_m.columns = ['PSNR', 'SSIM', 'MSE']

if os.path.isfile(paths_valid[0] + '\\' + model + '_psnr_ssim_mse.csv'):
    print(model + '_psnr_ssim_mse.csv 檔案存在, 不存檔')
else:
    df_Scores_p_s_m.to_csv(
        paths_valid[0] + '\\' + model + '_psnr_ssim_mse.csv')
    print(model + '_psnr_ssim_mse.csv 存檔成功')

# %%

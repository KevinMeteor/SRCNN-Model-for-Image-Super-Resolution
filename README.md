# SRCNN-Model-for-Image-Super-Resolution
This is a deep learning project applying the SRCNN model, proposed in the paper ['Image Super-Resolution Using Deep Convolutional Networks,'](https://arxiv.org/abs/1501.00092) and implemented with the PyTorch library.
Some code in this repository is from the article: 
[SRCNN Implementation in PyTorch for Image Super Resolution](https://debuggercafe.com/srcnn-implementation-in-pytorch-for-image-super-resolution/)
on cafe website.

I add *image_psnr_ssim_mse_func.py, psnr_ssim_mse.py, search_cuda_version.py*, and another 3 model architecture in *srcnn.py* in [src folder](https://github.com/KevinMeteor/SRCNN-Model-for-Image-Super-Resolution/blob/main/src).


## Datasets
Data are from 3 datasets: Set5, Set14, and T91. Both of Set5 and Set14 are for validation and testing. And T91 is for training.


## Installation 
If your device with CUDA and its version is greater than v11.8, you can consider creating a conda environment for all the dependencies by
```
conda env create --file torch_118.yaml --name torch_118
```


### Prepare Train Dataset
SRCNN uses patches for training, which are downscaled, upscaled, patchfied from the T91 dataset, to create a number of training data from a few numbers of images.
Firstly, we will use *patchify_image.py* , in which
```python
low_res_img = cv2.resize(patch, (int(w*0.5), int(h*0.5)), 
                                        interpolation=cv2.INTER_CUBIC)

```
0.5 is for getting 2x upsampled bicubic blurry images. We can set the number(<1.) to contral the upsampled rate. 

Excute the code in *patchify_image.py* from the terminal from the src directory:
```
python patchify_image.py
```
Images in 't91_hr_patches' are training labels, and 't91_lr_patches' are training input.


### Prepare Validation Dataset
For validation, we conbine the Set5 and Set14 datasets in 'original' folders, totaling 19 images, as the ground truth.
and the bicubic 2x upsampled low resolution images, as the input.

Secoundly, we use *bicubic.py*  to prepare the validation dataset, in which
```python
if args['scale_factor'] == '2x':
    scale_factor = 0.5
    os.makedirs('../input/test_bicubic_rgb_2x', exist_ok=True)
    save_path_lr = '../input/test_bicubic_rgb_2x'
    os.makedirs('../input/test_hr', exist_ok=True)
    save_path_hr = '../input/test_hr'
```
'2x' and 0.5 is the scaling-factor, defining the bicubic scale for downsampling and upscaling.

Execute the code in *bicubic.py* from the terminal from the src directory:
```
python bicubic.py --path ../input/Set14/original ../input/Set5/original --scale-factor 2x
```

**Now, we have competely prepared all needed data!**


## Usage


### Train and Validate
All models are saves in the *srcnn.py* file.

To train the model *SRCNN1* with a zoom factor of 2, for 100 epochs on GPU.
From **lines 59** in *train.py* to set the model we wanted to be trained:
```python
model = srcnn.SRCNN1().to(device)
```

Execute the code in *train.py* from the terminal from the src directory to starting:
```
python train.py --epochs 100   --weights ../outputs/model_SRCNN1_ckpt.pth
```

If you have a trained model, you can load the weights to resume training (optional).


### Test
Finally, we can compare the test PSNR on the Set5 and Set14 datasets.

Execute the code in *test.py* from the terminal from the src directory to starting:
```
python test.py
```

Besides, we can also compare the test PSNR, SSIM, and MSE on the Set5 and Set14 datasets.

Execute the code in *psnr_ssim_mse.py* from the terminal from the src directory to starting:
```
python psnr_ssim_mse.py
```
and notice the model name that you sets.


## Results List

### Loss & PPSR

1. SRCNN1 Model


|Train and Validation Loss|Train and Validation PPSR|
|--|--|
| <img src="https://github.com/KevinMeteor/SRCNN-Model-for-Image-Super-Resolution/blob/main/outputs/valid_results/SRCNN1_valid_results/loss.png"  width="450"> | <img src="https://github.com/KevinMeteor/SRCNN-Model-for-Image-Super-Resolution/blob/main/outputs/valid_results/SRCNN1_valid_results/psnr.png"  width="450"> |


2. SRCNN2 Model

|Train and Validation Loss|Train and Validation PPSR|
|--|--|
| <img src="https://github.com/KevinMeteor/SRCNN-Model-for-Image-Super-Resolution/blob/main/outputs/valid_results/SRCNN2_valid_results/loss.png"  width="450"> | <img src="https://github.com/KevinMeteor/SRCNN-Model-for-Image-Super-Resolution/blob/main/outputs/valid_results/SRCNN2_valid_results/psnr.png"  width="450"> |





3. SRCNN3

|Train and Validation Loss|Train and Validation PPSR|
|--|--|
| <img src="https://github.com/KevinMeteor/SRCNN-Model-for-Image-Super-Resolution/blob/main/outputs/valid_results/SRCNN3_valid_results/loss.png"  width="450"> | <img src="https://github.com/KevinMeteor/SRCNN-Model-for-Image-Super-Resolution/blob/main/outputs/valid_results/SRCNN3_valid_results/psnr.png"  width="450"> |






4. SRCNN4

|Train and Validation Loss|Train and Validation PPSR|
|--|--|
| <img src="https://github.com/KevinMeteor/SRCNN-Model-for-Image-Super-Resolution/blob/main/outputs/valid_results/SRCNN4_valid_results/loss.png"  width="450"> | <img src="https://github.com/KevinMeteor/SRCNN-Model-for-Image-Super-Resolution/blob/main/outputs/valid_results/SRCNN4_valid_results/psnr.png"  width="450"> |


### Test Results
Epoachs = 100.

1. SRCNN1 Model


|Original (PSNR, SSIM, MSE)|BICUBIC x2 (26.12 dB, 0.90, 159.01)|SRCNN x2 (27.08 dB, 0.92, 127.30)|
|--|--|--|
| <img src="https://github.com/KevinMeteor/SRCNN-Model-for-Image-Super-Resolution/blob/main/input/test_hr/butterfly.png"  width="250">| <img src="https://github.com/KevinMeteor/SRCNN-Model-for-Image-Super-Resolution/blob/main/input/test_bicubic_rgb_2x/butterfly.png"  width="250"> | <img src="https://github.com/KevinMeteor/SRCNN-Model-for-Image-Super-Resolution/blob/main/outputs/valid_results/SRCNN1_valid_results/val_sr_100_5.png"  width="250"> |

|Original (PSNR, SSIM, MSE)|BICUBIC x2 (29.39 dB, 0.92, 74.76)|SRCNN x2 (29.10 dB, 0.91, 79.92)|
|--|--|--|
| <img src="https://github.com/KevinMeteor/SRCNN-Model-for-Image-Super-Resolution/blob/main/input/test_hr/zebra.png"  width="250">| <img src="https://github.com/KevinMeteor/SRCNN-Model-for-Image-Super-Resolution/blob/main/input/test_bicubic_rgb_2x/zebra.png"  width="250"> | <img src="https://github.com/KevinMeteor/SRCNN-Model-for-Image-Super-Resolution/blob/main/outputs/valid_results/SRCNN1_valid_results/val_sr_100_18.png"  width="250"> |

|Original (PSNR, SSIM, MSE)|BICUBIC x2 (32.40 dB, 0.96, 37.42)|SRCNN x2 (30.90 dB, 0.96, 58.82)|
|--|--|--|
| <img src="https://github.com/KevinMeteor/SRCNN-Model-for-Image-Super-Resolution/blob/main/input/test_hr/lenna.png"  width="250">| <img src="https://github.com/KevinMeteor/SRCNN-Model-for-Image-Super-Resolution/blob/main/input/test_bicubic_rgb_2x/lenna.png"  width="250"> | <img src="https://github.com/KevinMeteor/SRCNN-Model-for-Image-Super-Resolution/blob/main/outputs/valid_results/SRCNN1_valid_results/val_sr_100_12.png"  width="250"> |

|Original (PSNR, SSIM, MSE)|BICUBIC x2 (25.58 dB, 0.94, 179.91)|SRCNN x2 (25.22 dB, 0.93, 195.55)|
|--|--|--|
| <img src="https://github.com/KevinMeteor/SRCNN-Model-for-Image-Super-Resolution/blob/main/input/test_hr/ppt3.png"  width="250">| <img src="https://github.com/KevinMeteor/SRCNN-Model-for-Image-Super-Resolution/blob/main/input/test_bicubic_rgb_2x/ppt3.png"  width="250"> | <img src="https://github.com/KevinMeteor/SRCNN-Model-for-Image-Super-Resolution/blob/main/outputs/valid_results/SRCNN1_valid_results/val_sr_100_16.png"  width="250"> |

## My Hardware Specs
```
Intel Core i7-12700 CPU 
NVIDIA GeForce RTX 3080 (32GB)
32GB DDR4 RAM
```

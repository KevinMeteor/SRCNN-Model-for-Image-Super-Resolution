# SRCNN-Model-for-Image-Super-Resolution
This is a deep learning project applying the SRCNN model, proposed in the paper ['Image Super-Resolution Using Deep Convolutional Networks,'](https://arxiv.org/abs/1501.00092) and implemented with the PyTorch library.
Some code in this repository is from an article: 
[SRCNN Implementation in PyTorch for Image Super Resolution](https://debuggercafe.com/srcnn-implementation-in-pytorch-for-image-super-resolution/)
on cafe website.

I add *image_psnr_ssim_mse_func.py, psnr_ssim_mse.py, search_cuda_version.py*, and another 3 model architecture in *srcnn.py* in [src folder](https://github.com/KevinMeteor/SRCNN-Model-for-Image-Super-Resolution/blob/main/src).


## Dataset
Data are from 3 dataset: Set5, Set14, and T91. Both of Set5 and Set14 are for validation and testing. And T91 is for training.


### Prepare Train Dataset
SRCNN uses patches for training, which are downscaled, upscaled, patchfied from the T91 dataset, to create the number(<1.) training data from a few numbers of images.
Firstly, we will use *patchify_iamge.py* , in which
```
low_res_img = cv2.resize(patch, (int(w*0.5), int(h*0.5)), 
                                        interpolation=cv2.INTER_CUBIC)

```
0.5 is to get 2x upsampled bicubic blurry images. We can set a number to contral the upsampled rate. 

Excute the code from the terminal:
```
python patchify_image.py
```
Images in 't91_hr_patches' are training labels, and 't91_lr_patches' are training inputs.


### Prepare Train Dataset



## Installation 
If your device with CUDA and its version is greater than v11.8, you can consider creating a conda environment for all the dependencies by
```
conda env create --file torch_118.yaml --name torch_118
```


## My Hardware Specs
```
Intel Core i7-12700 CPU 
NVIDIA GeForce RTX 3080 (32GB)
32GB DDR4 RAM
```

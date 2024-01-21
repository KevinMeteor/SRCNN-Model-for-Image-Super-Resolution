# SRCNN-Model-for-Image-Super-Resolution
This is a deep learning project applying the SRCNN model, proposed in the paper ['Image Super-Resolution Using Deep Convolutional Networks,'](https://arxiv.org/abs/1501.00092) and implemented with the PyTorch library.
Some code in this repository is from an article: 
[SRCNN Implementation in PyTorch for Image Super Resolution](https://debuggercafe.com/srcnn-implementation-in-pytorch-for-image-super-resolution/)
on cafe website.

I add image_psnr_ssim_mse_func.py, psnr_ssim_mse.py, search_cuda_version.py, and another 3 model architecture in srcnn.py in src folder.



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

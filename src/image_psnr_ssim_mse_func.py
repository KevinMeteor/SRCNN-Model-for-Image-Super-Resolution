

"""
20240120
Computing PSNR, SSIM, MSE between the Ground-truth image and the Test image or any two images.

https://scikit-image.org/docs/stable/api/skimage.metrics.html
-----------------------------------------------------
synax:
img1 = cv2.imread('Ground-truth image image.png')
img2 = cv2.imread('Test image image.png')

psnr = compare_psnr(img1, img2)
    Returns : 
        psnr : float
            The PSNR metric.

ssim = compare_ssim(img1, img2, channel_axis=None)
    Parameters :
        channel_axis : int or None, optional
            If None, the image is assumed to be a grayscale (single channel) image. 
            Otherwise, this parameter indicates which axis of the array corresponds to channels.
    Returns:
        mssim : float
            The mean structural similarity index over the image.


mse = compare_mse(img1, img2)
    Returns : 
        mse : float
            The mean-squared error (MSE) metric.


----------------------------------------------------
"""

# %%
from skimage.metrics import mean_squared_error as compare_mse
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity as compare_ssim
import cv2


def compare_images_scores(img1, img2, channel_axis=None):
    """
    Inputs : 
        img1 : ndarray
            Ground-truth image
        img2 : ndarray
            Test image
        channel_axis : int or None, optional
            If None, the image is assumed to be a grayscale (single channel) image. 
            Otherwise, this parameter indicates which axis of the array corresponds to channels.

    Returns:
        psnr : float (bigger is greater.)
                The PSNR metric.
        mssim : float ( < 1., bigger is greater.)
            The mean structural similarity index over the image.
        mse : float (bigger is worser.)
            The mean-squared error (MSE) metric.

    """
    p = compare_psnr(img1, img2)
    s = compare_ssim(img1, img2, channel_axis=channel_axis)
    m = compare_mse(img1, img2)

    return (p, s, m)


# %%
if __name__ == '__main__':
    path = '..'
    img1 = cv2.imread(
        path + '\\input\\test_hr\\butterfly.png')
    img2 = cv2.imread(
        path + '\\outputs\\valid_results\\SRCNN1_valid_results\\val_sr_100_5.png')

    p = compare_psnr(img1, img2)
    s = compare_ssim(img1, img2, channel_axis=2)
    m = compare_mse(img1, img2)

    print(f'PSNR = {p:.2f}')
    print(f'SSIM = {s:.2f}')
    print(f'MSE = {m:.2f}')

    A = compare_images_scores(img1, img2, 2)


# %%

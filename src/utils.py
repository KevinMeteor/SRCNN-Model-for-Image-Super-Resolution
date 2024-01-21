import math
import numpy as np
import matplotlib.pyplot as plt
import torch

from torchvision.utils import save_image

plt.style.use('ggplot')


def psnr(label, outputs, max_val=1.):
    """
    Compute Peak Signal to Noise Ratio (the higher the better).
    PSNR = 20 * log10(MAXp) - 10 * log10(MSE).
    https://en.wikipedia.org/wiki/Peak_signal-to-noise_ratio#Definition

    Note that the output and label pixels (when dealing with images) should
    be normalized as the `max_val` here is 1 and not 255.
    """
    label = label.cpu().detach().numpy()
    outputs = outputs.cpu().detach().numpy()
    diff = outputs - label
    rmse = math.sqrt(np.mean((diff) ** 2))
    if rmse == 0:
        return 100
    else:
        PSNR = 20 * math.log10(max_val / rmse)
        return PSNR


def save_plot(train_loss, val_loss, train_psnr, val_psnr):
    # Loss plots.
    plt.figure(figsize=(10, 7))
    plt.plot(train_loss, color='orange', label='train loss')
    plt.plot(val_loss, color='red', label='validataion loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('../outputs/loss.png')
    plt.close()

    # PSNR plots.
    plt.figure(figsize=(10, 7))
    plt.plot(train_psnr, color='green', label='train PSNR dB')
    plt.plot(val_psnr, color='blue', label='validataion PSNR dB')
    plt.xlabel('Epochs')
    plt.ylabel('PSNR (dB)')
    plt.legend()
    plt.savefig('../outputs/psnr.png')
    plt.close()


def save_model_state(model):
    # save the model to disk
    # (only trained weights, We can use this for inference and share this as well,
    # because this will be smaller in size)
    print('Saving model...')
    torch.save(model.state_dict(), '../outputs/model_SRCNN1.pth')


def save_model(epochs, model, optimizer, criterion):
    """
    Function to save the trained model to disk.
    """
    # Remove the last model checkpoint if present.
    # (saves the entire model checkpoint.
    #  This will be a larger model,
    #  because of the extra information but we can use this to resume training in the future if the case arises.)
    torch.save({
        'epoch': epochs+1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': criterion,
    }, f"../outputs/model_SRCNN1_ckpt.pth")


def save_validation_results(outputs, epoch, batch_iter):
    """
    Function to save the validation reconstructed images.
    """
    # ( We will save them every 500 epochs as per the training script.)

    save_image(
        outputs,
        f"../outputs/valid_results/val_sr_{epoch}_{batch_iter}.png"
    )

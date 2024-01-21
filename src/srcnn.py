import torch.nn as nn
import torch.nn.functional as F
from math import sqrt
import torch


class SRCNN1(nn.Module):
    def __init__(self):
        super(SRCNN1, self).__init__()

        self.conv1 = nn.Conv2d(
            3, 64, kernel_size=9, stride=(1, 1), padding=(2, 2)
        )
        self.conv2 = nn.Conv2d(
            64, 32, kernel_size=1, stride=(1, 1), padding=(2, 2)
        )
        self.conv3 = nn.Conv2d(
            32, 3, kernel_size=5, stride=(1, 1), padding=(2, 2)
        )

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.conv3(x)

        return x


class SRCNN2(nn.Module):
    def __init__(self):
        super(SRCNN2, self).__init__()

        self.conv1 = nn.Conv2d(
            # padding_mode='replicate'
            3, 128, kernel_size=9, stride=(1, 1), padding=(2, 2),
        )
        self.conv2 = nn.Conv2d(
            # padding_mode='replicate'
            128, 64, kernel_size=1, stride=(1, 1), padding=(2, 2),
        )
        self.conv3 = nn.Conv2d(
            # padding_mode='replicate'
            64, 3, kernel_size=5, stride=(1, 1), padding=(2, 2),
        )

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.conv3(x)

        return x


class SRCNN3(nn.Module):
    def __init__(self):
        super(SRCNN3, self).__init__()

        self.conv1 = nn.Conv2d(3, 128, kernel_size=9, padding=2)
        self.conv2 = nn.Conv2d(128, 64, kernel_size=1, padding=2)
        self.conv3 = nn.Conv2d(64, 32, kernel_size=1, padding=2)
        self.conv4 = nn.Conv2d(32, 64, kernel_size=5, padding=2)
        self.conv5 = nn.Conv2d(64, 3, kernel_size=5)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = self.conv5(x)

        return x

# the larger filter size SRCNN


class SRCNN4(nn.Module):
    """
    Section 5.4 of:
    https://personal.ie.cuhk.edu.hk/~ccloy/files/eccv_2014_deepresolution.pdf
    """

    def __init__(self):
        super(SRCNN4, self).__init__()

        # , padding_mode='replicate'
        self.conv1 = nn.Conv2d(3, 64, kernel_size=11, padding=3)
        self.conv2 = nn.Conv2d(64, 32, kernel_size=1, padding=3)
        self.conv3 = nn.Conv2d(32, 3, kernel_size=7, padding=2)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.conv3(x)

        return x


if __name__ == '__main__':
    model = SRCNN1()
    # Total parameters and trainable parameters.
    total_params = sum(p.numel() for p in model.parameters())
    print(f"{total_params:,} total parameters.")
    total_trainable_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad)
    print(f"{total_trainable_params:,} training parameters.")

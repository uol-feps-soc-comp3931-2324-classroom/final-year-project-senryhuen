import torch.nn as nn


class ConvGLU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride_size):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels * 2, kernel_size, stride_size, "same"
        )

    def forward(self, x):
        # conv
        x = self.conv(x)
        channels = x.shape[1]

        # glu
        x = x[:, : channels // 2, :, :] * x[:, channels // 2 :, :, :].sigmoid()
        return x


# DNN layers of ConvGlu (all stride_size of 1):
# convglu, input 6 channels, output 32, kernel (11,11)
# convglu, input 32 channels, output 32, kernel (7,3)
# convglu, input 32 channels, output 32, kernel (7,3)
# convglu, input 32 channels, output 32, kernel (7,3)
# conv, input 32 channels, output 2, kernel (7,3)
class DNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.first = nn.Sequential(ConvGLU(6, 32, 11, 1))

        self.middle = nn.Sequential(
            ConvGLU(32, 32, (7, 3), 1),
            ConvGLU(32, 32, (7, 3), 1),
        )

        self.last = nn.Sequential(
            ConvGLU(32, 32, (7, 3), 1),
            nn.Conv2d(32, 2, (7, 3), 1, "same"),
        )

    def forward(self, x):
        x = self.first(x)
        residual = x
        x = self.middle(x)
        x += residual
        x = self.last(x)
        return x

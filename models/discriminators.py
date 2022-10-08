import paddle as pd
import paddle.nn as nn
import numpy as np
from utils.functions import Downsample2X2

CHANNEL = 3

XH = [3,  3,   3,   3,   3,   3,   3,   3]
XK = [64, 128, 128, 256, 256, 512, 512, 512]


class DiscriminatorX(nn.Layer):
    """
    降采样图像的鉴别器，输入降采样的图像，输出该图是真实低分辨率图像的概率
    对应原来代码实现中的discriminator_X函数
    """

    # H = [ 3, 3, 3,  3,  3,  3,  3] # kernel size
    # K = [32,64,64,128,128,256,256] # number of feature map

    def __init__(self, image_size, H=XH, K=XK) -> None:
        super().__init__()
        assert len(H) == len(
            K
        ), f"The kernel size and the number of feature map must be the same length, got H:{len(H)} and K:{len(K)}"
        self.H = H
        self.K = K
        self.image_size = image_size
        self.feature_count = int(np.prod(image_size) / 64 * self.K[-1])

        # modify the H and K array for easier layer creation
        self.H = self.H
        self.K = [CHANNEL] + self.K

        self.layers = pd.nn.LayerList()

        # the conv2d+conv2d+downsampling layers
        layer_count = 0
        for h, (k_prev, k_next) in zip(self.H, zip(self.K[:-1], self.K[1:])):
            self.layers.append(
                nn.Conv2D(
                    in_channels=k_prev,
                    out_channels=k_next,
                    kernel_size=h,
                    stride=1,
                    padding="SAME"
                )
            )
            self.layers.append(nn.InstanceNorm2D(k_next))
            self.layers.append(nn.LeakyReLU(0.2))
            layer_count += 1
            if layer_count % 2 == 0 and 0 < layer_count < 8:
                self.layers.append(Downsample2X2())

        # FC1 and FC2 for the final output
        # first reshape the output of the last conv2d layer into [B, -1]
        self.layers.extend(
            [
                nn.Flatten(),
                nn.Linear(in_features=self.feature_count, out_features=256),
                nn.LeakyReLU(0.2),
                nn.Linear(in_features=256, out_features=1),
            ]
        )
        # self.init_weight()

    def forward(self, x: pd.Tensor) -> pd.Tensor:
        for layer in self.layers:
            x = layer(x)
            # print(layer, x.shape)
        return x


YH = [3,  3,   3,   3,   3,   3,   3,   3]
YK = [64, 128, 128, 256, 256, 512, 512, 512]


class DiscriminatorY(pd.nn.Layer):
    """
    超分图像的鉴别器，输入超分的图像，输出该图是真实高分辨率图像的概率
    对应原来代码实现中的discriminator_Y函数
    """

    def __init__(self, image_size, H=YH, K=YK) -> None:
        super().__init__()

        assert len(H) == len(
            K
        ), f"The kernel size and the number of feature map must be the same length, got H:{len(H)} and K:{len(K)}"
        self.H = H
        self.K = K
        self.image_size = image_size

        # modify the H and K array for easier layer creation
        self.H = self.H
        self.K = [CHANNEL] + self.K

        # conv2d+conv2d+downsampling layers
        self.layers = pd.nn.LayerList()
        layer_count = 0
        for h, (k_prev, k_next) in zip(self.H, zip(self.K[:-1], self.K[1:])):
            self.layers.append(
                nn.Conv2D(
                    in_channels=k_prev,
                    out_channels=k_next,
                    kernel_size=h,
                    stride=1,
                    padding="SAME"
                )
            )
            self.layers.append(nn.InstanceNorm2D(k_next))
            self.layers.append(nn.LeakyReLU(0.2))
            layer_count += 1
            if layer_count % 2 == 0 and 0 < layer_count < 10:
                self.layers.append(Downsample2X2())

        # FC layers for the final output
        # first reshape the output of the last conv2d layer into [B, -1]
        self.layers.extend(
            [
                nn.Flatten(),
                nn.Linear(
                    in_features=int(np.prod(image_size) / 256 * self.K[-1]),
                    out_features=256,
                ),
                nn.LeakyReLU(0.2),
                nn.Linear(in_features=256, out_features=1),
            ]
        )

    def forward(self, x: pd.Tensor) -> pd.Tensor:
        for layer in self.layers:
            x = layer(x)
            # print(layer, x.shape)
        return x


if __name__ == "__main__":
    B, W, H, C = 16, 128, 128, 3
    image = pd.rand([B, C, W, H])
    # average pool image to mimic low res
    low_res_image = pd.nn.functional.avg_pool2d(image, kernel_size=4, stride=4)
    DX = DiscriminatorX([W/4, H/4])
    results = DX(low_res_image)
    print(results)

    DY = DiscriminatorY([W, H])
    results = DY(image)
    print(results)

import paddle as pd
import paddle.nn as nn
import numpy as np
from utils.functions import Downsample2X2, Upsample2X2

GH = [3,   3,   3,   3,   3,   3,   3,   3,   3]
GK = [128, 256, 256, 512, 512, 512, 256, 256, 3]

CHANNEL = 3

class ResidualBlock(nn.Layer):
    def __init__(self, layers):
        super().__init__()
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return x * 0.5 + self.model(x)


class GeneratorG(nn.Layer):
    """
    低分辨率湍流图像到高分辨率湍流图像的生成器
    从原论文上看，输出的结果是[128,128]，输入是128/4, /8, /16
    """

    def __init__(self, image_size, H=GH, K=GK, ratio=4, start_2x=2) -> None:
        super().__init__()
        assert len(H) == len(
            K
        ), f"The kernel size and the number of feature map must be the same length, got H:{len(H)} and K:{len(K)}"
        self.H = H
        self.K = [CHANNEL] + K
        self.image_size = image_size

        # modify the H and K array for easier layer creation
        self.H = self.H
        self.K = self.K + [CHANNEL]
        self.ratio = 4
        self.start_2x = start_2x
        self.end_2x = start_2x + self.ratio + 2

        self.layers = pd.nn.LayerList()

        # the conv2d+conv2d+upsampling layers
        layer_count = 0
        for h, (k_prev, k_next) in zip(self.H, zip(self.K[:-1], self.K[1:])):
            if k_prev == k_next:
                self.layers.append(
                    ResidualBlock([
                    nn.Conv2D(
                        in_channels=k_prev,
                        out_channels=k_next,
                        kernel_size=h,
                        stride=1,
                        padding=h//2,
                        padding_mode="reflect"
                    ),
                    nn.LeakyReLU(0.2)]))
            else:
                self.layers.append(nn.Conv2D(in_channels=k_prev, out_channels = k_next, kernel_size=h, stride=1, padding=h//2, padding_mode="reflect"))
                self.layers.append(nn.LeakyReLU(0.2))
            layer_count += 1
            if layer_count % 2 == 0 and self.start_2x < layer_count < self.end_2x:
                self.layers.append(Upsample2X2(2, 2))
        
        self.layers.append(nn.Tanh())

    def forward(self, x: pd.Tensor) -> pd.Tensor:
        for layer in self.layers:
            x = layer(x)
            # print(layer, x.shape)
        return x


FH = [3,  3,  3,   3,   3,   3,   3,   3,   3,   3]
FK = [32, 64, 128, 128, 256, 256, 512, 512, 512, 3]


class GeneratorF(nn.Layer):
    """
    高分辨率到低分辨率湍流图像的生成器
    """

    def __init__(self, image_size, H=FH, K=FK, ratio=4, start_2x=2):
        super().__init__()
        assert len(H) == len(
            K
        ), f"The kernel size and the number of feature map must be the same length, got H:{len(H)} and K:{len(K)}"
        self.H = H
        self.K = [CHANNEL] + K
        self.image_size = image_size
        self.ratio = ratio
        self.start_2x = start_2x
        self.end_2x = self.ratio + self.start_2x + 2

        self.layers = pd.nn.LayerList()

        # the conv2d+conv2d+upsampling layers
        layer_count = 0
        for h, (k_prev, k_next) in zip(self.H, zip(self.K[:-1], self.K[1:])):
            if k_prev == k_next:
                self.layers.append(
                    ResidualBlock([
                    nn.Conv2D(
                        in_channels=k_prev,
                        out_channels=k_next,
                        kernel_size=h,
                        stride=1,
                        padding=h//2,
                        padding_mode="reflect"
                    ),
                    nn.LeakyReLU(0.2)
                    ])
                )
            else:
                self.layers.append(nn.Conv2D(in_channels=k_prev, out_channels=k_next, kernel_size=h, stride=1, padding=h//2, padding_mode="reflect"))
                self.layers.append(nn.LeakyReLU(0.2))
            layer_count += 1
            if layer_count % 2 == 0 and self.start_2x < layer_count < self.end_2x:
                self.layers.append(Downsample2X2())
        self.layers.append(nn.Tanh())

    def forward(self, x: pd.Tensor) -> pd.Tensor:
        for layer in self.layers:
            x = layer(x)
            # print(layer, x.shape)
        return x


if __name__ == "__main__":
    B, C, H, W = 1, 3, 32, 32
    x = pd.randn([B, C, H, W])
    model = GeneratorG([H, W])
    y = model(x)
    print(y.shape) # [1, 3, 128, 128]

    model2 = GeneratorF([H * 4, W * 4])
    x = model2(y)
    print(x.shape) # [1, 3, 32, 32]

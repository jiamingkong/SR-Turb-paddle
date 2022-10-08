import paddle as pd
import numpy as np


def get_init_range(layer: pd.nn.Layer, gain=np.sqrt(2)):
    """
    获取权重初始化范围 (mu, std)
    """
    shape = layer.weight.shape
    fan_in = np.prod(shape[:-1])
    std = gain / np.sqrt(fan_in)
    return (0, std)


def upsampling(image: pd.Tensor, p: int = 2, q: int = 2):
    """
    图像的超采样前置转置操作:
    输入 image: paddle.Tensor([B, W, H, C]),
            p: int,宽度放大系数
            q: int,高度放大系数
    输出 image_tiled: paddle.Tensor([B, W*p, H*q, C])
    """
    B, C, H, W = image.shape
    # Add two dimensions to A for tiling
    image_exp = pd.reshape(image, [B, C, H, 1, W, 1])
    # Tile image along new dimensions
    image_tiled = pd.tile(image_exp, [1, 1, 1, p, 1, q])
    # Reshape
    image_tiled = pd.reshape(image_tiled, [B, C, H * p, W * q])
    return image_tiled


def downsample2X2(x: pd.Tensor):
    """
    图像降采样，宽度和高度变成原来的1/2
    """
    return (
        pd.add_n(
            [
                x[:, :, 0::2, 0::2],
                x[:, :, 1::2, 0::2],
                x[:, :, 0::2, 1::2],
                x[:, :, 1::2, 1::2],
            ]
        )
        / 4.0
    )


class Downsample2X2(pd.nn.Layer):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: pd.Tensor):
        return (
            pd.add_n(
                [
                    x[:, :, 0::2, 0::2],
                    x[:, :, 1::2, 0::2],
                    x[:, :, 0::2, 1::2],
                    x[:, :, 1::2, 1::2],
                ]
            )
            / 4.0
        )


class Upsample2X2(pd.nn.Layer):
    def __init__(self, p=2, q=2) -> None:
        super().__init__()
        self.p = p
        self.q = q

    def forward(self, image: pd.Tensor):
        B, C, H, W = image.shape
        # Add two dimensions to A for tiling
        image_exp = pd.reshape(image, [B, C, H, 1, W, 1])
        # Tile image along new dimensions
        image_tiled = pd.tile(image_exp, [1, 1, 1, self.p, 1, self.q])
        # Reshape
        image_tiled = pd.reshape(image_tiled, [B, C, H * self.p, W * self.q])
        return image_tiled


if __name__ == "__main__":
    B, W, H, C = 16, 256, 256, 3
    image = pd.rand([B, C, W, H])
    image_tiled = upsampling(image, p=2, q=2)
    print(image_tiled.shape)  # [16,512,512,3]
    image_downsample = downsample2X2(image)
    print(image_downsample.shape)  # [16,128,128,3]

    ds2 = Downsample2X2()
    image_downsample = ds2(image)
    print(image_downsample.shape)  # [16,128,128,3]

    us2 = Upsample2X2()
    image_upsample = us2(image)
    print(image_upsample.shape)  # [16,512,512,3]

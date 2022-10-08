import paddle as pd
import paddle.nn as nn


def identity_loss(image_A: pd.Tensor, image_B: pd.Tensor):
    return pd.mean(pd.abs(image_A - image_B))

def mse_loss(image_A: pd.Tensor, image_B: pd.Tensor):
    return pd.mean(pd.square(image_A - image_B))


def wgp_slope_condition(gen_dict: dict, discX: nn.Layer, discY: nn.Layer) -> tuple:
    """
    给定生成器生成的数据，使用discriminatorX和discriminatorY完成对应的鉴别步骤
    """
    # uniform of 0 to 1
    batch = gen_dict["X_real"].shape[0]
    alpha = pd.rand([batch, 1, 1, 1])
    # interpolate between real and fake
    X_hat = (alpha * gen_dict["X_real"] + (1.0 - alpha) * gen_dict["X_predict"])
    Y_hat = (alpha * gen_dict["Y_real"] + (1.0 - alpha) * gen_dict["Y_predict"])
    # calculate gradient
    X_grad = pd.grad(
        outputs=discX(X_hat), inputs=X_hat, create_graph=False, retain_graph=True
    )[0]
    slopes_X = pd.sqrt(pd.mean(pd.square(X_grad), axis=[1, 2, 3]))
    Y_grad = pd.grad(
        outputs=discY(Y_hat), inputs=Y_hat, create_graph=False, retain_graph=True
    )[0]
    slopes_Y = pd.sqrt(pd.mean(pd.square(Y_grad), axis=[1, 2, 3]))
    # calculate gradient penalty
    gradient_penalty_X = pd.mean(pd.square(slopes_X - 1.0))
    gradient_penalty_Y = pd.mean(pd.square(slopes_Y - 1.0))
    return {
        "gradient_penalty_X": gradient_penalty_X,
        "gradient_penalty_Y": gradient_penalty_Y,
    }


def cycle_consistency_loss(gen_dict: dict, lambda_cycle: float) -> tuple:
    """
    给定生成器生成的数据，使用discriminatorX和discriminatorY完成对应的鉴别步骤
    """
    # gen_dict["X_cycle"].shape = B, C, H, W = 16, 3, 32, 32
    # X_cycle_loss = pd.mean(pd.abs(gen_dict["X_cycle"] - gen_dict["X_real"]))
    X_cycle_loss = mse_loss(gen_dict["X_cycle"], gen_dict["X_real"]) + identity_loss(gen_dict["X_cycle"], gen_dict["X_real"])
    # Y_cycle_loss = pd.mean(pd.abs(gen_dict["Y_cycle"] - gen_dict["Y_real"]))
    Y_cycle_loss = mse_loss(gen_dict["Y_cycle"], gen_dict["Y_real"]) + identity_loss(gen_dict["Y_cycle"], gen_dict["Y_real"])
    return {"X_cycle_loss": X_cycle_loss, "Y_cycle_loss": Y_cycle_loss}

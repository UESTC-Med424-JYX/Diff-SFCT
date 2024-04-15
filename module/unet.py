import math
from typing import Optional, Sequence, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from monai.networks.blocks import Convolution, UpSample
from monai.networks.layers.factories import Conv, Pool
from monai.utils import deprecated_arg, ensure_tuple_rep

from module.attention import AttentionBlock
from module.SFCT import Spatial_Frequency_Cross_Transformer

image_size = 224


def get_timestep_embedding(timesteps, embedding_dim):
    assert len(timesteps.shape) == 1

    half_dim = embedding_dim // 2
    emb = math.log(10000) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float32) * -emb)
    emb = emb.to(device=timesteps.device)
    emb = timesteps.float()[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if embedding_dim % 2 == 1:  # zero pad
        emb = torch.nn.functional.pad(emb, (0, 1, 0, 0))
    return emb


# swish
def nonlinearity(x):
    return x * torch.sigmoid(x)


class ResBlock(nn.Sequential):

    @deprecated_arg(name="dim", new_name="spatial_dims", since="0.6", msg_suffix="Please use `spatial_dims` instead.")
    def __init__(
            self,
            spatial_dims: int,
            in_chns: int,
            out_chns: int,
            act: Union[str, tuple],
            norm: Union[str, tuple],
            bias: bool,
            dropout: Union[float, tuple] = 0.0,
            dim: Optional[int] = None,
    ):
        super().__init__()
        self.temb_proj = torch.nn.Linear(512, out_chns)

        if dim is not None:
            spatial_dims = dim

        self.spatial_dims = spatial_dims

        conv_0 = Convolution(spatial_dims, in_chns, out_chns, act=act, norm=norm, dropout=dropout, bias=bias, padding=1)
        conv_1 = Convolution(
            spatial_dims, out_chns, out_chns, act=act, norm=norm, dropout=dropout, bias=bias, padding=1
        )
        self.add_module("conv_0", conv_0)
        self.add_module("conv_1", conv_1)

        if in_chns == out_chns:
            self.skip_connection = nn.Identity()
        else:
            self.skip_connection = Convolution(
                spatial_dims, in_chns, out_chns, act=act, norm=norm, dropout=dropout, bias=bias, padding=1
            )

    def forward(self, x, temb=None):
        h = self.conv_0(x)
        if temb is not None:
            if self.spatial_dims == 3:
                h = h + self.temb_proj(nonlinearity(temb))[:, :, None, None, None]
            elif self.spatial_dims == 2:
                h = h + self.temb_proj(nonlinearity(temb))[:, :, None, None]
        h = self.conv_1(h)
        return self.skip_connection(x) + h


class Down(nn.Sequential):

    @deprecated_arg(name="dim", new_name="spatial_dims", since="0.6", msg_suffix="Please use `spatial_dims` instead.")
    def __init__(
            self,
            spatial_dims: int,
            in_chns: int,
            out_chns: int,
            act: Union[str, tuple],
            norm: Union[str, tuple],
            bias: bool,
            dropout: Union[float, tuple] = 0.0,
            dim: Optional[int] = None,
    ):
        super().__init__()
        if dim is not None:
            spatial_dims = dim
        max_pooling = Pool["MAX", spatial_dims](kernel_size=2)
        convs = ResBlock(spatial_dims, in_chns, out_chns, act, norm, bias, dropout)
        self.add_module("max_pooling", max_pooling)
        self.add_module("convs", convs)

    def forward(self, x, temb=None):
        x = self.max_pooling(x)
        x = self.convs(x, temb)
        return x


class UpCat(nn.Module):

    @deprecated_arg(name="dim", new_name="spatial_dims", since="0.6", msg_suffix="Please use `spatial_dims` instead.")
    def __init__(
            self,
            spatial_dims: int,
            in_chns: int,
            cat_chns: int,
            out_chns: int,
            act: Union[str, tuple],
            norm: Union[str, tuple],
            bias: bool,
            dropout: Union[float, tuple] = 0.0,
            upsample: str = "deconv",
            pre_conv: Optional[Union[nn.Module, str]] = "default",
            interp_mode: str = "linear",
            align_corners: Optional[bool] = True,
            halves: bool = True,
            dim: Optional[int] = None,
    ):

        super().__init__()
        if dim is not None:
            spatial_dims = dim
        if upsample == "nontrainable" and pre_conv is None:
            up_chns = in_chns
        else:
            up_chns = in_chns // 2 if halves else in_chns
        self.upsample = UpSample(
            spatial_dims,
            in_chns,
            up_chns,
            2,
            mode=upsample,
            pre_conv=pre_conv,
            interp_mode=interp_mode,
            align_corners=align_corners,
        )
        self.convs = ResBlock(spatial_dims, cat_chns + up_chns, out_chns, act, norm, bias, dropout)

    def forward(self, x: torch.Tensor, x_e: Optional[torch.Tensor], temb=None):
        x_0 = self.upsample(x)

        if x_e is not None:
            dimensions = len(x.shape) - 2
            sp = [0] * (dimensions * 2)
            for i in range(dimensions):
                if x_e.shape[-i - 1] != x_0.shape[-i - 1]:
                    sp[i * 2 + 1] = 1
            x_0 = torch.nn.functional.pad(x_0, sp, "replicate")
            x = self.convs(torch.cat([x_e, x_0], dim=1), temb)
        else:
            x = self.convs(x_0, temb)

        return x


class Input_Block(nn.Sequential):

    @deprecated_arg(name="dim", new_name="spatial_dims", since="0.6", msg_suffix="Please use `spatial_dims` instead.")
    def __init__(
            self,
            spatial_dims: int,
            in_chns: int,
            out_chns: int,
            act: Union[str, tuple],
            norm: Union[str, tuple],
            bias: bool,
            dropout: Union[float, tuple] = 0.0,
    ):
        super().__init__()

        self.conv = ResBlock(spatial_dims, in_chns, in_chns, act, norm, bias, dropout)
        self.attention = AttentionBlock(in_chns, image_size)
        self.down = Down(spatial_dims, in_chns, out_chns, act, norm, bias, dropout)

    def forward(self, x, temb=None):
        x = self.conv(x, temb)
        x = self.attention(x)
        x = self.down(x, temb)
        return x


class Bottle_Neck(nn.Module):

    @deprecated_arg(name="dim", new_name="spatial_dims", since="0.6", msg_suffix="Please use `spatial_dims` instead.")
    def __init__(
            self,
            spatial_dims: int,
            in_chns: int,
            out_chns: int,
            act: Union[str, tuple],
            norm: Union[str, tuple],
            bias: bool,
            dropout: Union[float, tuple] = 0.0,
            img_dim=16,
            head_num=4,
            mlp_dim=512,
            block_num=12,
            patch_dim=1
    ):
        super().__init__()

        self.conv_1 = ResBlock(spatial_dims, in_chns, in_chns, act, norm, bias, dropout)
        self.conv_2 = ResBlock(spatial_dims, in_chns, in_chns, act, norm, bias, dropout)

        self.fsct = Spatial_Frequency_Cross_Transformer(img_dim=img_dim,
                                                        in_channels=in_chns,
                                                        embedding_dim=out_chns,
                                                        head_num=head_num,
                                                        mlp_dim=mlp_dim,
                                                        block_num=block_num,
                                                        patch_dim=patch_dim)

        self.conv_3 = ResBlock(spatial_dims, in_chns, out_chns, act, norm, bias, dropout)

    '''
    x : frequency
    y : spatial
    '''

    def forward(self, x, y, temb=None):
        x = self.conv_1(x, temb)

        y = self.conv_2(y)
        x = self.fsct(x, y, temb)

        x = self.conv_3(x, temb)
        return x


class Output_Block(nn.Module):

    @deprecated_arg(name="dim", new_name="spatial_dims", since="0.6", msg_suffix="Please use `spatial_dims` instead.")
    def __init__(
            self,
            spatial_dims: int,
            in_chns: int,
            cat_chns: int,
            out_chns: int,
            act: Union[str, tuple],
            norm: Union[str, tuple],
            bias: bool,
            dropout: Union[float, tuple] = 0.0,
            upsample: str = "deconv",
            is_out=False,
    ):
        super().__init__()

        self.up = UpCat(spatial_dims, in_chns, cat_chns, out_chns, act, norm, bias, dropout, upsample)
        self.attention = AttentionBlock(out_chns, image_size, is_out)
        self.conv = ResBlock(spatial_dims, out_chns, out_chns, act, norm, bias, dropout)

    def forward(self, x: torch.Tensor, x_e: Optional[torch.Tensor], temb=None):
        x = self.up(x, x_e, temb)
        x = self.attention(x)
        x = self.conv(x, temb)
        return x


class Diff_SFCT(nn.Module):
    @deprecated_arg(
        name="dimensions", new_name="spatial_dims", since="0.6", msg_suffix="Please use `spatial_dims` instead."
    )
    def __init__(
            self,
            spatial_dims: int = 3,
            num_modality: int = 1,
            num_classes: int = 3,
            features: Sequence[int] = (32, 32, 64, 128, 256, 32),
            act: Union[str, tuple] = ("LeakyReLU", {"negative_slope": 0.1, "inplace": True}),
            norm: Union[str, tuple] = ("instance", {"affine": True}),
            bias: bool = True,
            dropout: Union[float, tuple] = 0.0,
            upsample: str = "deconv",
            dimensions: Optional[int] = None,
    ):
        super().__init__()
        if dimensions is not None:
            spatial_dims = dimensions

        fea = ensure_tuple_rep(features, 6)
        print(f"BasicUNet features: {fea}.")

        # timestep embedding
        self.temb = nn.Module()
        self.temb.dense = nn.ModuleList([
            torch.nn.Linear(128, 512),
            torch.nn.Linear(512, 512),
        ])

        '''
        input_blocks
        '''
        self.in_conv = ResBlock(spatial_dims, num_modality + num_classes, features[0], act, norm, bias, dropout)
        self.input_blocks = nn.ModuleList([
            Input_Block(spatial_dims, fea[0], fea[1], act, norm, bias, dropout),
            Input_Block(spatial_dims, fea[1], fea[2], act, norm, bias, dropout),
            Input_Block(spatial_dims, fea[2], fea[3], act, norm, bias, dropout),
            Input_Block(spatial_dims, fea[3], fea[4], act, norm, bias, dropout),
        ])

        '''
        Bottleneck
        '''
        self.bottle_neck = Bottle_Neck(spatial_dims, fea[4], fea[4], act, norm, bias, dropout,
                                       img_dim=14,
                                       head_num=4,
                                       mlp_dim=512,
                                       block_num=6,
                                       patch_dim=1)

        '''
        output_blocks
        '''
        self.output_blocks = nn.ModuleList([
            Output_Block(spatial_dims, fea[4], fea[3], fea[3], act, norm, bias, dropout, upsample),
            Output_Block(spatial_dims, fea[3], fea[2], fea[2], act, norm, bias, dropout, upsample),
            Output_Block(spatial_dims, fea[2], fea[1], fea[1], act, norm, bias, dropout, upsample),
            Output_Block(spatial_dims, fea[1], fea[0], fea[5], act, norm, bias, dropout, upsample, is_out=True),
        ])

        '''
        image 分支
        '''
        self.image_in_conv = ResBlock(spatial_dims, num_modality, features[0], act, norm, bias, dropout)
        self.feature_encoder = nn.ModuleList([
            Input_Block(spatial_dims, fea[0], fea[1], act, norm, bias, dropout),
            Input_Block(spatial_dims, fea[1], fea[2], act, norm, bias, dropout),
            Input_Block(spatial_dims, fea[2], fea[3], act, norm, bias, dropout),
            Input_Block(spatial_dims, fea[3], fea[4], act, norm, bias, dropout),
        ])

        self.final_conv = Conv["conv", spatial_dims](fea[5], num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor, t, image):
        temb = get_timestep_embedding(t, 128)
        temb = self.temb.dense[0](temb)
        temb = nonlinearity(temb)
        temb = self.temb.dense[1](temb)

        x = torch.cat([x, image], dim=1)

        x = self.in_conv(x, temb)
        y = self.image_in_conv(image)
        x += y

        h = []

        for module_x, module_y in zip(self.input_blocks, self.feature_encoder):
            h.append(x)
            x = module_x(x)
            y = module_y(y)
            x += y

        x = self.bottle_neck(x, y, temb)

        for module in self.output_blocks:
            x = module(x, h.pop(), temb)

        logits = self.final_conv(x)

        return logits


if __name__ == '__main__':
    x = torch.rand(1, 3, 224, 224)
    image = torch.rand(1, 1, 224, 224)
    t = torch.zeros([1])

    num_modality = 1
    num_classes = 3

    model = Diff_SFCT(2, num_modality, num_classes, [64, 128, 256, 512, 1024, 128],
                      act=("LeakyReLU", {"negative_slope": 0.1, "inplace": False}))

    out = model(x, t, image)

    print(out.shape)
